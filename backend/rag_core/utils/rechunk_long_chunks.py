"""
Re-chunk over-length passages into tokenizer-friendly segments.

주요 단계
1. 전처리: 공백/개행 정규화
2. 문단 및 문장 단위 분할
3. 토큰 길이를 기준으로 재조합 (목표 토큰 수 이하)
4. JSONL로 결과 저장 (각 항목에 parent_chunk_id 포함)

예시:
  python utils/rechunk_long_chunks.py \
      --chunk-file 2_Chunking/output/all_chunks.json \
      --long-chunks PJ/Chat/eval/long_chunks.jsonl \
      --tokenizer BAAI/bge-reranker-base \
      --target-length 380 \
      --max-length 512 \
      --output PJ/Chat/eval/rechunked_long_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-chunk long corpus segments into shorter pieces")
    parser.add_argument(
        "--chunk-file",
        default=str(Path(__file__).resolve().parents[1] / "2_Chunking/output/all_chunks.json"),
        help="전체 청크 JSON 경로",
    )
    parser.add_argument(
        "--long-chunks",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/long_chunks.jsonl"),
        help="장문 청크 JSONL 경로 (export_long_chunks.py 출력)",
    )
    parser.add_argument(
        "--tokenizer",
        default="BAAI/bge-reranker-base",
        help="토크나이저 이름 또는 경로",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=380,
        help="청크 재조합 시 목표 토큰 수",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="절대 허용 최대 토큰 수",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/rechunked_long_chunks.jsonl"),
        help="결과를 저장할 JSONL 경로",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {entry["chunk_id"]: entry for entry in data}


def iter_long_chunks(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_units(text: str) -> List[str]:
    """문단 → 문장 단위로 분할."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []
    sentence_pattern = re.compile(r".+?(?:[.!?]|다\.|요\.)", re.S)

    for para in paragraphs:
        sentences = [s.strip() for s in sentence_pattern.findall(para)]
        if not sentences:
            sentences = [para]
        units.extend(sentences)

    return [unit for unit in units if unit]


def token_length(tokenizer: AutoTokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def split_by_tokens(tokenizer: AutoTokenizer, text: str, max_length: int) -> List[str]:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    if len(token_ids) <= max_length:
        return [text.strip()]

    chunks: List[str] = []
    for idx in range(0, len(token_ids), max_length):
        sub_ids = token_ids[idx : idx + max_length]
        decoded = tokenizer.decode(sub_ids, skip_special_tokens=True)
        chunks.append(decoded.strip())
    return chunks


def regroup_units(
    tokenizer: AutoTokenizer,
    units: List[str],
    target_length: int,
    max_length: int,
) -> List[str]:
    results: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = token_length(tokenizer, unit)

        if unit_tokens > max_length:
            if current_units:
                results.append("\n\n".join(current_units).strip())
                current_units = []
                current_tokens = 0
            results.extend(split_by_tokens(tokenizer, unit, max_length))
            continue

        if current_units and current_tokens + unit_tokens > target_length:
            results.append("\n\n".join(current_units).strip())
            current_units = [unit]
            current_tokens = unit_tokens
        else:
            current_units.append(unit)
            current_tokens += unit_tokens

    if current_units:
        results.append("\n\n".join(current_units).strip())

    # 마지막 방어: 여전히 max_length 초과하는 항목은 토큰 기반으로 재분할
    final_chunks: List[str] = []
    for candidate in results:
        if token_length(tokenizer, candidate) <= max_length:
            final_chunks.append(candidate)
        else:
            final_chunks.extend(split_by_tokens(tokenizer, candidate, max_length))
    return final_chunks


def main() -> None:
    args = parse_args()

    chunk_map = load_chunks(Path(args.chunk_file))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e6)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_original = 0
    total_new = 0

    with output_path.open("w", encoding="utf-8") as fp:
        for entry in iter_long_chunks(Path(args.long_chunks)):
            chunk_id = entry["chunk_id"]
            original = chunk_map.get(chunk_id, {})
            source = original.get("source")
            raw_text = entry.get("text") or original.get("text") or ""

            total_original += 1

            normalized = normalize_text(raw_text)
            units = split_into_units(normalized)
            if not units:
                continue

            new_chunks = regroup_units(tokenizer, units, args.target_length, args.max_length)

            for idx, text_part in enumerate(new_chunks, start=1):
                new_chunk_id = f"{chunk_id}_part{idx:02d}"
                payload = {
                    "chunk_id": new_chunk_id,
                    "parent_chunk_id": chunk_id,
                    "source": source,
                    "text": text_part,
                }
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
                total_new += 1

    print(
        f"{total_original}개 청크를 재분할하여 {total_new}개의 새 청크를 {output_path}에 저장했습니다."
    )


if __name__ == "__main__":
    main()
