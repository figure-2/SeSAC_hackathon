"""
Export list of long chunks that exceed a token threshold.

사용법:
  python utils/export_long_chunks.py \
      --chunk-file 2_Chunking/output/all_chunks.json \
      --tokenizer BAAI/bge-reranker-base \
      --max-length 512 \
      --output PJ/Chat/eval/long_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chunk metadata that exceed max token length")
    parser.add_argument(
        "--chunk-file",
        default=str(Path(__file__).resolve().parents[1] / "2_Chunking/output/all_chunks.json"),
        help="청크 데이터(JSON) 경로",
    )
    parser.add_argument(
        "--tokenizer",
        default="BAAI/bge-reranker-base",
        help="토큰 길이 측정을 위한 HuggingFace 토크나이저 이름 또는 경로",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="허용 가능한 최대 토큰 길이 (기본: 512)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/long_chunks.jsonl"),
        help="결과를 저장할 JSONL 경로",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure: expected list, got {type(data).__name__}")
    return data


def main() -> None:
    args = parse_args()

    chunk_path = Path(args.chunk_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunk_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e6)

    exported = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for entry in chunks:
            text = entry.get("text") or ""
            chunk_id = entry.get("chunk_id") or "(unknown)"
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            token_len = len(token_ids)
            if token_len <= args.max_length:
                continue

            payload = {
                "chunk_id": chunk_id,
                "token_length": token_len,
                "text": text,
            }
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            exported += 1

    print(f"총 {exported}개의 장문 청크를 {output_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
