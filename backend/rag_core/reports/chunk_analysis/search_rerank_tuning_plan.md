# 검색·리랭크 파라미터 튜닝 계획 (2025-11-04 기준)

## 1. 목표
- `retrieve_k`, `rerank_k`, (추후) BM25 가중치를 조정해 Hit@k, MRR, nDCG를 향상시키고
- LLM에 전달되는 컨텍스트 품질을 개선하여 정성 평가 품질도 상승시키는 것.

## 2. 변경 사항 요약
- `run_batch_evaluation.py`에 `--retrieve-k`, `--rerank-k` 인자를 추가하여 초기 검색·리랭크 크기를 제어 가능.
- `run_batch_rag.py`에도 동일 인자를 추가하여 정성 평가 실행 시 동일한 조건을 적용 가능.
- 실행 시 `--run-name`에 파라미터를 명시(예: `retrieve30_rerank5`)하면 결과 폴더/리포트가 자동으로 구분된다.

## 3. 추천 실험 매트릭스
1. **`retrieve_k` 탐색 (`rerank_k=5`, `top_k=5` 고정)**  
   | 실험 | 커맨드 | 비고 |
   | --- | --- | --- |
   | A | `--retrieve-k 20 --rerank-k 5 --top-k 5` | 기본값 대비 후보 축소 |
   | B | `--retrieve-k 30 --rerank-k 5 --top-k 5` | 현재 권장 |
   | C | `--retrieve-k 50 --rerank-k 5 --top-k 5` | 더 넓은 후보 확보 |
2. **`rerank_k` 탐색 (`retrieve_k` 최적값 사용)**  
   | 실험 | 커맨드 | 비고 |
   | --- | --- | --- |
   | D | `--retrieve-k 30 --rerank-k 3 --top-k 3` | LLM 컨텍스트 축소 |
   | E | `--retrieve-k 30 --rerank-k 5 --top-k 5` | 기준 |
   | F | `--retrieve-k 30 --rerank-k 8 --top-k 5` | LLM 입력 확장 |

> 각 실험마다 `run_batch_evaluation.py` → 정량 지표 확인, `run_batch_rag.py` → 정성 응답 확인.

## 4. 결과 정리 템플릿
- `7_Evaluation/output/<timestamp>_<run-name>/summary.md`의 Aggregated Metrics를 `evaluation_history.json`과 함께 비교.
- `6_Integrated_RAG/results/<timestamp>_<run-name>/summary.md` + `answers/*.md`를 빠르게 훑어보고 반복/헛소리 여부 체크.
- 최종 비교는 `PJ/Chat/eval/search_rerank_tuning_results.md`와 같은 문서에 기록.

## 5. 향후 확장
- BM25 가중치, RRF 파라미터 등 하이브리드 검색 튜닝 시에도 동일 구조로 실험할 수 있도록 스크립트 확장.
- 정량 평가 커맨드에 `--bm25-weight`, `--rrf-k` 등을 추가해 추후 분석 자동화 준비.
