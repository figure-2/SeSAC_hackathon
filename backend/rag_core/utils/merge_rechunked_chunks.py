"""
Merge re-chunked segments back into the corpus without overwriting originals.

Example:
  python utils/merge_rechunked_chunks.py \
      --chunk-file 2_Chunking/output/all_chunks.json \
      --long-chunks PJ/Chat/eval/long_chunks.jsonl \
      --rechunked PJ/Chat/eval/rechunked_long_chunks.jsonl \
      --output PJ/Chat/eval/all_chunks_rechunked.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge re-chunked data with original corpus.")
    parser.add_argument(
        "--chunk-file",
        default=str(Path(__file__).resolve().parents[1] / "2_Chunking/output/all_chunks.json"),
        help="원본 청크 JSON 경로",
    )
    parser.add_argument(
        "--long-chunks",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/long_chunks.jsonl"),
        help="재분할 대상 장문 청크 목록(JSONL)",
    )
    parser.add_argument(
        "--rechunked",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/rechunked_long_chunks.jsonl"),
        help="재분할된 청크(JSONL)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/all_chunks_rechunked.json"),
        help="병합된 청크를 저장할 JSON 경로",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    args = parse_args()

    chunk_path = Path(args.chunk_file)
    long_path = Path(args.long_chunks)
    rechunked_path = Path(args.rechunked)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_chunks = load_json(chunk_path)
    long_chunks = load_jsonl(long_path)
    rechunked_chunks = load_jsonl(rechunked_path)

    original_map: Dict[str, Dict] = {item["chunk_id"]: item for item in original_chunks}
    long_ids: Set[str] = {item["chunk_id"] for item in long_chunks}

    merged: List[Dict] = []
    skipped = 0
    for chunk in original_chunks:
        if chunk["chunk_id"] in long_ids:
            skipped += 1
            continue
        merged.append(chunk)

    parent_source_cache: Dict[str, str] = {}
    for chunk in rechunked_chunks:
        parent_id = chunk.get("parent_chunk_id")
        source = chunk.get("source")
        if not source and parent_id:
            if parent_id not in parent_source_cache:
                parent = original_map.get(parent_id, {})
                parent_source_cache[parent_id] = parent.get("source")
            source = parent_source_cache.get(parent_id)

        merged.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": source or "unknown",
                "text": chunk["text"],
            }
        )

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(merged, fp, ensure_ascii=False, indent=2)

    print(f"원본 청크: {len(original_chunks)}")
    print(f"재분할 제거: {skipped} / 추가: {len(rechunked_chunks)}")
    print(f"결과 청크: {len(merged)} (저장 위치: {output_path})")


if __name__ == "__main__":
    main()
