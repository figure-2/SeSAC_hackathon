"""
Chunk length analyzer.

This utility inspects the chunked corpus and reports token length statistics.
It highlights chunks that exceed the typical CrossEncoder / retriever limit
(default: 512 tokens) so that we can prioritise re-chunking or summarisation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse token lengths of chunked corpus")
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
        "--top-n",
        type=int,
        default=20,
        help="가장 긴 청크 상위 N개를 보고합니다 (기본: 20)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "PJ/Chat/eval/chunk_length_report.md"),
        help="결과를 저장할 Markdown 경로",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[Dict]:
    """Load chunk data from either JSON array or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        chunks: List[Dict] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure: expected list, got {type(data).__name__}")
    return data


def percentile(sorted_lengths: Sequence[int], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not sorted_lengths:
        return 0.0
    if p <= 0:
        return float(sorted_lengths[0])
    if p >= 100:
        return float(sorted_lengths[-1])
    k = (len(sorted_lengths) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_lengths[int(k)])
    d0 = sorted_lengths[f] * (c - k)
    d1 = sorted_lengths[c] * (k - f)
    return float(d0 + d1)


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, float]:
    if not lengths:
        return {}
    sorted_lengths = sorted(lengths)
    total = len(sorted_lengths)
    return {
        "count": total,
        "min": float(sorted_lengths[0]),
        "median": percentile(sorted_lengths, 50),
        "mean": sum(sorted_lengths) / total,
        "p75": percentile(sorted_lengths, 75),
        "p90": percentile(sorted_lengths, 90),
        "p95": percentile(sorted_lengths, 95),
        "max": float(sorted_lengths[-1]),
    }


def render_report(
    summary: Dict[str, float],
    over_limit_stats: Dict[str, float],
    length_distribution: Counter,
    offenders: List[Tuple[str, int, str]],
    over_limit_prefixes: Counter,
    max_length: int,
) -> str:
    lines = [
        "# Chunk Token Length Report",
        "",
        f"- 총 청크 수: {int(summary.get('count', 0))}",
        f"- 평균 토큰 수: {summary.get('mean', 0):.2f}",
        f"- 중앙값 토큰 수: {summary.get('median', 0):.2f}",
        "",
        "## 분포 (토큰 수)",
        "| 지표 | 값 |",
        "| --- | --- |",
        f"| 최소 | {summary.get('min', 0):.0f} |",
        f"| 75 분위 | {summary.get('p75', 0):.0f} |",
        f"| 90 분위 | {summary.get('p90', 0):.0f} |",
        f"| 95 분위 | {summary.get('p95', 0):.0f} |",
        f"| 최대 | {summary.get('max', 0):.0f} |",
        "",
    ]

    over_count = int(over_limit_stats.get("count", 0))
    over_ratio = over_limit_stats.get("ratio", 0.0) * 100
    lines.extend(
        [
            "## 임계값 초과 현황",
            f"- 임계값: {max_length} 토큰",
            f"- 초과 청크 수: {over_count} ({over_ratio:.2f}% of total)",
            f"- 초과 청크 평균 길이: {over_limit_stats.get('mean', 0):.2f}",
            f"- 초과 청크 최대 길이: {over_limit_stats.get('max', 0):.0f}",
            "",
        ]
    )

    if offenders:
        lines.extend(
            [
                "## 최상위 초과 청크",
                "| 순위 | chunk_id | 토큰 수 | 본문 앞 80자 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for idx, (chunk_id, token_len, text) in enumerate(offenders, start=1):
            preview = text.replace("\n", " ")[:80]
            lines.append(f"| {idx} | `{chunk_id}` | {token_len} | {preview} |")
        lines.append("")

    # 간단한 히스토그램(토큰 구간)
    if length_distribution:
        lines.extend(
            [
                "## 토큰 길이 구간 (100 토큰 단위)",
                "| 구간 | 청크 수 |",
                "| --- | --- |",
            ]
        )
        for bucket in sorted(length_distribution.keys()):
            lines.append(f"| {bucket} | {length_distribution[bucket]} |")
        lines.append("")

    if over_limit_prefixes:
        lines.extend(
            [
                "## 임계 초과 청크 상위 Prefix",
                "| prefix | 청크 수 |",
                "| --- | --- |",
            ]
        )
        for prefix, count in over_limit_prefixes.most_common(10):
            lines.append(f"| {prefix} | {count} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    chunk_path = Path(args.chunk_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunk_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # 모델 최대 길이 제한을 넉넉하게 늘려 경고를 방지합니다.
    tokenizer.model_max_length = int(1e6)

    lengths: List[int] = []
    offenders: List[Tuple[str, int, str]] = []
    bucket_counter: Counter = Counter()
    prefix_counter: Counter = Counter()

    for entry in chunks:
        text = entry.get("text") or ""
        chunk_id = entry.get("chunk_id") or "(unknown)"
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
        token_len = len(token_ids)
        lengths.append(token_len)

        if token_len > args.max_length:
            offenders.append((chunk_id, token_len, text))
            prefix = chunk_id.split("_", 1)[0]
            prefix_counter[prefix] += 1

        bucket = f"{(token_len // 100) * 100:04d}-{((token_len // 100) * 100) + 99:04d}"
        bucket_counter[bucket] += 1

    summary = summarize_lengths(lengths)

    over_limit_lengths = [length for length in lengths if length > args.max_length]
    over_limit_stats = {
        "count": len(over_limit_lengths),
        "ratio": len(over_limit_lengths) / len(lengths) if lengths else 0.0,
        "mean": sum(over_limit_lengths) / len(over_limit_lengths) if over_limit_lengths else 0.0,
        "max": max(over_limit_lengths) if over_limit_lengths else 0,
    }

    offenders.sort(key=lambda item: item[1], reverse=True)
    top_offenders = offenders[: args.top_n]

    report = render_report(
        summary,
        over_limit_stats,
        bucket_counter,
        top_offenders,
        prefix_counter,
        args.max_length,
    )
    output_path.write_text(report, encoding="utf-8")

    print(f"분석 완료: {output_path}")


if __name__ == "__main__":
    main()
