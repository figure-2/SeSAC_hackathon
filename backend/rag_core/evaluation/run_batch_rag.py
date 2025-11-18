"""
Qualitative RAG batch runner skeleton.

대표 질문 세트를 순차 실행하여 RAG 파이프라인 출력과
사용된 근거 문서를 기록하는 템플릿입니다.
`run_single_query` 함수에 실제 검색/생성 함수를 연결하세요.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import os

import chromadb
import yaml
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from utils.config import DEFAULT_CONFIG_PATH, load_config

PROMPT_STYLES = ("baseline", "cot", "citation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative RAG batch runner (skeleton)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="env.yaml 경로 (기본값: utils.config.DEFAULT_CONFIG_PATH)",
    )
    parser.add_argument(
        "--queries",
        default="PJ/Chat/eval/queries.yaml",
        help="정성 평가에 사용할 질의 YAML 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="6_Integrated_RAG/results",
        help="실행 로그를 저장할 상위 디렉터리",
    )
    parser.add_argument(
        "--run-name",
        help="출력 폴더 이름에 사용할 접두사 (예: nightly_regression)",
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
        default=5,
        help="리랭커 상위 결과 중 LLM에 전달할 문서 수 (기본: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="평가할 질의 수를 제한하고 싶을 때 사용",
    )
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="baseline",
        help="LLM 프롬프트 스타일 (baseline, cot, citation)",
    )
    parser.add_argument(
        "--llm-model",
        help="env.yaml 설정 대신 사용할 Gemini 모델 이름",
    )
    return parser.parse_args()


def load_queries(path: Path) -> List[Dict]:
    """YAML에서 질의 목록을 로드하고 `qualitative` 플래그를 기준으로 필터링합니다."""
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}

    queries = payload.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"유효한 질의를 찾지 못했습니다: {path}")

    filtered = [item for item in queries if item.get("qualitative", True)]
    if not filtered:
        raise ValueError("qualitative용 질의가 비어 있습니다. YAML에 `qualitative: true`를 지정하세요.")
    return filtered


def prepare_output_dir(base_dir: Path, run_name: str | None) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}" if not run_name else f"{timestamp}_{run_name}"
    root = base_dir / folder_name
    root.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "meta": root / "meta.json",
        "summary": root / "summary.md",
        "answers_dir": root / "answers",
    }


def create_prompt(question: str, context_docs: List[str], style: str) -> str:
    """프롬프트 스타일에 따라 명령문을 구성합니다."""
    contexts = "\n\n".join(context_docs)
    if style == "cot":
        instruction = (
            "당신은 역사 전문가 AI입니다. 아래 제공된 '참고 자료'만을 활용하여 질문에 답변하세요. "
            "먼저 참고 자료를 검토하며 사고 과정을 bullet 형식으로 정리한 뒤, 마지막에 최종 답변을 제시하세요. "
            "사고 과정에는 참고한 자료 번호를 포함하고, 최종 답변은 간결하게 요약하세요."
        )
        output_format = "## 사고 과정\n- 단계별로 생각을 정리하세요.\n\n## 최종 답변"
    elif style == "citation":
        instruction = (
            "당신은 역사 전문가 AI입니다. 아래 '참고 자료'에 등장하는 정보만 사용하여 질문에 답변하세요. "
            "각 문장 끝에는 참고한 자료 번호를 [자료 n] 형식으로 명시해야 합니다. "
            "자료에 없는 내용은 추론하지 말고, 답변은 간결한 문단 형식으로 작성하세요."
        )
        output_format = "[답변]"
    else:
        instruction = (
            "당신은 역사적 사실에 기반하여 질문에 답변하는 역사 전문가 AI입니다. "
            "반드시 아래 제공된 '참고 자료'만을 활용하여 질문에 대한 답변을 생성해야 합니다. "
            "'참고 자료'에 언급되지 않은 내용은 절대로 답변에 포함해서는 안 됩니다. "
            "답변은 한국어로 작성해주세요."
        )
        output_format = "[답변]"

    prompt = f"""{instruction}

[질문]
{question}

[참고 자료]
{contexts}

[출력 형식]
{output_format}
"""
    return prompt


def run_single_query(
    query_payload: Dict,
    config: Dict,
    embeddings_model: HuggingFaceEmbeddings,
    collection: chromadb.Collection,
    reranker_model: CrossEncoder,
    llm,
    retrieve_k: int,
    rerank_k: int,
    prompt_style: str,
) -> Dict:
    """
    단일 질의에 대해 RAG 파이프라인을 실행하고 로그를 반환합니다.
    """
    question = query_payload["question"]
    
    # 1. Retrieval
    query_embedding = embeddings_model.embed_query(question)
    retrieval_results = collection.query(query_embeddings=[query_embedding], n_results=retrieve_k)
    retrieved_ids = retrieval_results['ids'][0]
    retrieved_docs = retrieval_results['documents'][0]

    # 2. Reranking
    reranked_docs_with_scores = sorted(
        zip(reranker_model.predict([[question, doc] for doc in retrieved_docs]), 
            retrieved_ids, 
            retrieved_docs),
        key=lambda x: x[0],
        reverse=True,
    )
    
    final_docs = [doc for score, doc_id, doc in reranked_docs_with_scores[:rerank_k]]
    final_doc_ids = [doc_id for score, doc_id, doc in reranked_docs_with_scores[:rerank_k]]
    
    # 3. Generation
    prompt = create_prompt(question, final_docs, prompt_style)
    try:
        response = llm.generate_content(prompt)
        answer = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
    except Exception as e:
        print(f"오류: Gemini API 호출 중 에러 발생 - {e}")
        answer = "[Gemini API 호출 오류]"
        usage = None

    return {
        "id": query_payload.get("id"),
        "question": question,
        "answer": answer.strip(),
        "contexts": final_docs,
        "context_ids": final_doc_ids,
        "retrieved_ids_initial": retrieved_ids,
        "params": {"retrieve_k": retrieve_k, "rerank_k": rerank_k},
        "usage_metadata": usage,
    }


def write_answer_file(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {payload.get('id', 'unknown id')}",
        "",
        f"**질문**: {payload.get('question', '')}",
        "",
        "## 답변",
        payload.get("answer", "(no answer)"),
        "",
        "## 참고 문서",
    ]
    for idx, context in enumerate(payload.get("contexts", []), start=1):
        lines.extend([f"- 문서 {idx}", "", f"```\n{context}\n```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, results: Iterable[Dict], meta: Dict) -> None:
    lines = [
        "# RAG Qualitative Batch Summary",
        "",
        f"- 실행 시각: {meta.get('timestamp')}",
        f"- 질의 수: {meta.get('num_queries')}",
        f"- 프롬프트 스타일: {meta.get('prompt_style')}",
        f"- LLM 모델: {meta.get('llm_model')}",
    ]

    usage = meta.get("usage")
    if usage:
        lines.extend(
            [
                f"- 총 프롬프트 토큰: {usage.get('total_prompt_tokens', 0)}",
                f"- 총 응답 토큰: {usage.get('total_response_tokens', 0)}",
            ]
        )

    lines.extend(
        [
            "",
            "| id | question | truncated_answer |",
            "| --- | --- | --- |",
        ]
    )

    for item in results:
        answer = (item.get("answer") or "").replace("\n", " ")
        truncated = (answer[:60] + "...") if len(answer) > 60 else answer
        lines.append(f"| {item.get('id')} | {item.get('question')} | {truncated} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    queries_path = Path(args.queries)
    queries = load_queries(queries_path)
    if args.limit:
        queries = queries[: args.limit]

    # Gemini API 환경 변수 로드
    api_env_path = config.get("LLM_API_ENV_PATH")
    if api_env_path:
        env_path = Path(api_env_path)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            if "GOOGLE_AI_STUDIO_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
        else:
            raise FileNotFoundError(f"LLM_API_ENV_PATH에 지정된 파일을 찾을 수 없습니다: {api_env_path}")

    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("환경 변수 GOOGLE_API_KEY가 설정되어 있지 않습니다. Gemini API 키를 확인하세요.")

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model_name = args.llm_model or config.get("GEMINI_MODEL_NAME", "models/gemini-pro")
    try:
        generation_config = {"temperature": 0 if args.prompt_style != "cot" else 0.2}
        llm = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    except Exception as e:
        print(f"오류: Gemini 모델({model_name}) 초기화 실패 - {e}")
        return

    # Retriever/Reranker 초기화
    print("RAG 파이프라인을 초기화합니다...")
    embedding_model_name = config["FINETUNED_EMBEDDING_MODEL"]
    reranker_model_name = config["FINETUNED_RERANKER_MODEL"]
    collection_name = config["FINETUNED_COLLECTION_NAME"]

    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cuda'})
    reranker_model = CrossEncoder(reranker_model_name, max_length=512, device='cuda')

    db_path = project_root / config["VECTOR_DB_DIR"] / collection_name
    if not Path(str(db_path)).exists():
         raise FileNotFoundError(f"Chroma DB 컬렉션을 찾을 수 없습니다: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=collection_name)
    print("초기화 완료.")

    outputs = prepare_output_dir(Path(args.output_dir), args.run_name)
    results: List[Dict] = []
    total_prompt_tokens = 0
    total_response_tokens = 0

    for query in tqdm(queries, desc="Batch RAG"):
        record = run_single_query(
            query,
            config,
            embeddings_model,
            collection,
            reranker_model,
            llm,
            args.retrieve_k,
            args.rerank_k,
            args.prompt_style,
        )
        results.append(record) # record에 이미 id와 question이 포함되어 있음

        answer_path = outputs["answers_dir"] / f"{query.get('id', 'result')}.md"
        write_answer_file(answer_path, record)

        usage = record.get("usage_metadata")
        if usage:
            total_prompt_tokens += usage.get("prompt_token_count", 0)
            total_response_tokens += usage.get("candidates_token_count", 0)

    meta_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(args.config).resolve()),
        "queries_path": str(queries_path.resolve()),
        "num_queries": len(results),
        "llm_model": model_name,
        "prompt_style": args.prompt_style,
        "parameters": {
            "retrieve_k": args.retrieve_k,
            "rerank_k": args.rerank_k,
        },
        "usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
        },
    }
    outputs["meta"].write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(outputs["summary"], results, meta_payload)

    print(f"\n정성 평가 결과가 저장되었습니다: {outputs['root']}")


if __name__ == "__main__":
    main()
