"""
Batch evaluation runner skeleton.

이 스크립트는 대표 질의 세트를 읽어 정량 지표를 일괄 계산하고
실행 결과를 타임스탬프별 폴더에 저장하는 골격 코드입니다.
실제 평가 로직은 `run_single_query` 함수 내부에 구현하세요.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb
import numpy as np
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.config import DEFAULT_CONFIG_PATH, ensure_keys, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation runner (skeleton)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="env.yaml 경로 (기본값: utils.config.DEFAULT_CONFIG_PATH)",
    )
    parser.add_argument(
        "--queries",
        default="PJ/Chat/eval/queries.yaml",
        help="평가 질의가 정의된 YAML 경로",
    )
    parser.add_argument(
        "--eval-dataset",
        help="JSON 포맷 평가 데이터셋 경로 (예: 7_Evaluation/eval_qa_dataset.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="7_Evaluation/output",
        help="평가 결과를 저장할 상위 디렉터리",
    )
    parser.add_argument(
        "--run-name",
        help="결과 폴더 이름에 사용할 커스텀 접두사 (예: finetuned_vs_baseline)",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=50,
        help="초기 벡터 검색에서 가져올 문서 수 (기본: 50)",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=10,
        help="리랭킹 후 고려할 문서 수 (기본: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval 결과에서 고려할 top-k (세부 지표 계산 시 사용)",
    )
    return parser.parse_args()


def load_queries_from_yaml(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    queries = payload.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"유효한 질의 목록을 찾지 못했습니다: {path}")
    return queries


def load_queries_from_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        queries = data.get("queries")
    else:
        queries = data
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"유효한 질의 목록을 찾지 못했습니다: {path}")
    required_keys = {"question", "ground_truth_context_id"}
    for idx, item in enumerate(queries):
        if not required_keys.issubset(item.keys()):
            raise ValueError(f"항목 {idx}에 필수 키가 없습니다: {required_keys}")
    return queries


def build_output_paths(base_dir: Path, run_name: str | None) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}" if not run_name else f"{timestamp}_{run_name}"
    root_dir = base_dir / folder_name
    root_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_dir,
        "metrics": root_dir / "metrics.json",
        "details_csv": root_dir / "retrieval_detail.csv",
        "meta": root_dir / "meta.json",
        "summary": root_dir / "summary.md",
    }


def calculate_metrics(retrieved_ids: List[str], ground_truth_id: str, top_k: int) -> Dict[str, float]:
    """단일 K값에 대해 Hit Rate, MRR, nDCG를 계산합니다."""
    metrics = {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    
    # K값에 맞춰 결과 슬라이싱
    retrieved_ids_k = retrieved_ids[:top_k]

    if ground_truth_id in retrieved_ids_k:
        rank = retrieved_ids_k.index(ground_truth_id) + 1
        metrics["hit_rate"] = 1.0
        metrics["mrr"] = 1.0 / rank

        true_relevance = np.zeros(len(retrieved_ids_k))
        true_relevance[rank - 1] = 1
        scores = np.arange(len(retrieved_ids_k), 0, -1)
        if true_relevance.sum() > 0:
            metrics["ndcg"] = ndcg_score([true_relevance], [scores])
    
    return metrics


def run_single_query(
    query_payload: Dict,
    embeddings_model: HuggingFaceEmbeddings,
    collection: chromadb.Collection,
    reranker_model: CrossEncoder,
    retrieve_k: int,
    rerank_k: int,
    top_k: int,
) -> Tuple[Dict, Dict]:
    """
    단일 질의에 대한 평가를 수행합니다.

    반환값:
        metrics: {"hit_rate": float, "mrr": float, "ndcg": float, ...}
        detail: {"id": str, "question": str, "retrieved_ids": [...], ...}
    """
    question = query_payload["question"]
    ground_truth_id = query_payload["ground_truth_context_id"]

    # 1. Retrieval
    query_embedding = embeddings_model.embed_query(question)
    # 리랭킹과 nDCG 계산을 위해 충분한 수(e.g., 50)를 검색
    retrieval_results = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
    retrieved_ids = retrieval_results['ids'][0]
    retrieved_docs = retrieval_results['documents'][0]

    # 2. Reranking
    reranked_docs_with_scores = sorted(
        zip(reranker_model.predict([[question, doc] for doc in retrieved_docs]), retrieved_ids),
        key=lambda x: x[0],
        reverse=True,
    )
    reranked_ids = [doc_id for score, doc_id in reranked_docs_with_scores[:rerank_k]]

    # 3. Calculate Metrics
    retrieval_metrics = calculate_metrics(retrieved_ids, ground_truth_id, top_k)
    rerank_metrics = calculate_metrics(reranked_ids, ground_truth_id, top_k)

    metrics = {
        f"retrieval_{key}": value for key, value in retrieval_metrics.items()
    }
    metrics.update({
        f"rerank_{key}": value for key, value in rerank_metrics.items()
    })

    detail = {
        "question": question,
        "ground_truth_id": ground_truth_id,
        "retrieved_ids": retrieved_ids[:top_k],
        "reranked_ids": reranked_ids[:top_k],
        "retrieve_k": retrieve_k,
        "rerank_k": rerank_k,
    }
    
    return metrics, detail


def aggregate_metrics(metrics_list: Iterable[Dict]) -> Dict:
    """개별 질의 지표를 평균 내어 전체 결과를 반환합니다."""
    totals: Dict[str, float] = {}
    count = 0
    for metrics in metrics_list:
        count += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def write_detail_csv(path: Path, detail_rows: Iterable[Dict]) -> None:
    rows = list(detail_rows)
    if not rows:
        return

    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, aggregate: Dict, metrics_list: List[Dict], output_meta: Dict) -> None:
    lines = [
        "# Batch Evaluation Summary",
        "",
        f"- 실행 시각: {output_meta.get('timestamp')}",
        f"- 모델 버전 정보: {output_meta.get('model_versions')}",
        "",
        "## Aggregated Metrics",
    ]
    for key, value in aggregate.items():
        lines.append(f"- **{key}**: {value:.4f}")

    lines.extend(
        [
            "",
            "## Per-query Metrics",
            "| id | retrieval_hit_rate | rerank_hit_rate | retrieval_mrr | rerank_mrr |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for metrics in metrics_list:
        lines.append(
            f"| {metrics.get('id')} "
            f"| {metrics.get('retrieval_hit_rate', 0.0):.4f} "
            f"| {metrics.get('rerank_hit_rate', 0.0):.4f} "
            f"| {metrics.get('retrieval_mrr', 0.0):.4f} "
            f"| {metrics.get('rerank_mrr', 0.0):.4f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    ensure_keys(config, ["VECTOR_DB_DIR", "FINETUNED_EMBEDDING_MODEL", "FINETUNED_RERANKER_MODEL"])

    if args.eval_dataset:
        dataset_path = Path(args.eval_dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"평가 데이터셋을 찾을 수 없습니다: {dataset_path}")
        queries = load_queries_from_json(dataset_path)
        queries_path = dataset_path
    else:
        queries_path = Path(args.queries)
        if not queries_path.exists():
            raise FileNotFoundError(f"YAML 질의 파일을 찾을 수 없습니다: {queries_path}")
        queries = load_queries_from_yaml(queries_path)

    output_paths = build_output_paths(Path(args.output_dir), args.run_name)

    # 모델 및 DB 클라이언트 초기화 (반복문 밖에서 한 번만)
    print("모델과 DB 클라이언트를 초기화합니다...")
    # TODO: env.yaml에서 어떤 모델(베이스라인/파인튜닝)을 쓸지 선택하는 로직 추가 필요
    embedding_model_name = config["FINETUNED_EMBEDDING_MODEL"]
    reranker_model_name = config["FINETUNED_RERANKER_MODEL"]
    collection_name = config["FINETUNED_COLLECTION_NAME"]

    embeddings_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'}, # 필요시 'cuda'로 변경
        encode_kwargs={'normalize_embeddings': True}
    )
    reranker_model = CrossEncoder(reranker_model_name, max_length=512)
    
    db_path = project_root / config["VECTOR_DB_DIR"] / collection_name
    if not db_path.exists():
        raise FileNotFoundError(f"Chroma DB 컬렉션을 찾을 수 없습니다: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=collection_name)
    print("초기화 완료.")

    meta_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(args.config).resolve()),
        "queries_path": str(queries_path.resolve()),
        "model_versions": {
            "baseline_embedding": config.get("BASELINE_EMBEDDING_MODEL"),
            "finetuned_embedding": config.get("FINETUNED_EMBEDDING_MODEL"),
            "baseline_reranker": config.get("BASELINE_RERANKER_MODEL"),
            "finetuned_reranker": config.get("FINETUNED_RERANKER_MODEL"),
        },
        "parameters": {
            "retrieve_k": args.retrieve_k,
            "rerank_k": args.rerank_k,
            "metrics_top_k": args.top_k,
        },
    }

    metrics_per_query: List[Dict] = []
    detail_rows: List[Dict] = []

    for query in tqdm(queries, desc="Batch Evaluation"):
        metrics, detail = run_single_query(
            query,
            embeddings_model,
            collection,
            reranker_model,
            args.retrieve_k,
            args.rerank_k,
            args.top_k,
        )
        metrics_per_query.append({"id": query.get("id"), **metrics})
        detail_rows.append({"id": query.get("id"), **detail})

    numeric_metrics = [
        {k: v for k, v in metrics.items() if k != 'id'}
        for metrics in metrics_per_query
    ]
    aggregate = aggregate_metrics(numeric_metrics)

    output_paths["metrics"].write_text(
        json.dumps({"aggregate": aggregate, "per_query": metrics_per_query}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_detail_csv(output_paths["details_csv"], detail_rows)
    output_paths["meta"].write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output_paths["summary"], aggregate, metrics_per_query, meta_payload)

    print(f"\n정량 평가 결과가 저장되었습니다: {output_paths['root']}")


if __name__ == "__main__":
    main()
