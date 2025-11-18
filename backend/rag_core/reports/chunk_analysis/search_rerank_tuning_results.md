# 검색·리랭크 튜닝 결과 기록 (2025-11-04)

## 1. 대표 질의 세트 (4문항) 실험
- `retrieve_k = {20, 30, 50}`, `rerank_k = 5`, `top_k = 5`
- 결과: 세 조합 모두 동일한 지표(히트율 0.25, 리랭크 히트율 0.50 등)를 출력
- 원인: 대표 세트가 단 4문항이라 파라미터 변화가 결과에 미치는 영향이 드러나지 않음

## 2. 공식 평가 데이터셋으로 확장
- `run_batch_evaluation.py` 에 `--eval-dataset` 인자를 추가하여 JSON(383문항)을 직접 평가 가능
- 실행 예시:
  ```bash
  python 7_Evaluation/run_batch_evaluation.py \
      --eval-dataset 7_Evaluation/eval_qa_dataset.json \
      --retrieve-k 30 --rerank-k 5 --top-k 5 \
      --run-name full_retrieve30_rerank5
  ```
- 향후 `retrieve_k`, `rerank_k`, (추후) BM25 가중치 등을 변경하며 반복 실행, 결과는 `summary.md`와 `evaluation_history.json` 비교

## 3. 정성 평가 연동
- `run_batch_rag.py`에도 동일 파라미터 인자를 추가했으므로, 정량·정성 모두 동일 조건에서 검증 가능
- 실행 예시:
  ```bash
  python 6_Integrated_RAG/run_batch_rag.py \
      --retrieve-k 30 --rerank-k 5 \
      --run-name full_retrieve30_rerank5
  ```
- 답변과 참고 문서를 `results/<timestamp>/answers/*.md`에서 검토

## 4. 다음 단계
- `retrieve_k`/`rerank_k` 조합 탐색
- BM25/RRF 파라미터 추가 후 탐색
- 정성평가 결과에 따라 프롬프트 및 LLM 후처리 개선
