# 긴 청크 조정 실행 계획

최근 토큰 길이 분석 결과(`utils/analyze_chunk_lengths.py`)에 따르면 전체 1,924개 청크 중 **607개(31.55%)**가 512 토큰을 초과했습니다.  
특히 `권력편`, `고려편`, `조선편`, `사건편`, `인물편`, `영웅편`, `근현대편`에서 초과 건수가 집중되어 있으며, 최댓값은 785 토큰입니다.

## 문제 유형 진단
- **연대기/전쟁사 서술**: 사건 흐름을 한 덩어리에 담은 서술형 콘텐츠 (`고려편`, `권력편`, `영웅편` 등).
- **사건 기록·법령**: 인용과 설명이 혼합되어 문단이 길게 이어지는 사례 (`사건편`, `근현대편`).
- **노이즈 다량 포함**: 공백·제어문자 등 불필요한 문자가 반복되는 청크 (`조선편_2025_00000` 등).

## 정리 전략
1. **전처리 개선**
   - 반복 공백, 제어문자, 의미 없는 패턴(`\n`, 특수기호 나열)을 우선 제거.
   - 문단 경계를 인식하기 쉬운 형태(2개 이상의 개행 → 단일 개행 등)로 정규화.

2. **의미 단위 청킹(1차)**
   - 각 문단을 `sentencepiece`/`kss` 등 문장 분할기로 분해 후,  
     문장 단위로 누적하며 400~450 토큰을 넘지 않도록 묶기.
   - 문단 사이에 요약문(2~3문장) 삽입하여 맥락 유지.

3. **요약 기반 압축(보조)**
   - 1차 청킹 이후에도 512 토큰을 넘는 경우, `t5-base-korean-summarization` 등 경량 모델로  
     2~3문장 요약을 추가 생성 후 본문과 교체.

4. **카테고리별 우선순위**
   - ① `권력편`, `고려편` (각 100건 이상)  
   - ② `조선편`, `사건편`, `인물편` (80~95건 수준)  
   - ③ `영웅편`, `근현대편` (60~70건 수준)

## 구현 로드맵
1. **청크 선택**  
   - `chunk_length_report.md` 목록을 기반으로 512 초과 청크 ID를 CSV/JSON으로 추출.  
     → `utils/export_long_chunks.py` 실행 시 `PJ/Chat/eval/long_chunks.jsonl` 생성 (607건)
2. **재청킹 스크립트 작성**  
   - 입력 텍스트 → 전처리 → 문장 분할 → 토큰 길이 계산 → 새 청크 생성 → JSONL 저장.  
     → `utils/rechunk_long_chunks.py` 실행 시 `PJ/Chat/eval/rechunked_long_chunks.jsonl` 생성 (1,214건)
3. **데이터 교체 & 증분 임베딩**  
   - 청크 파일 업데이트 후, 해당 청크만 재임베딩하여 벡터 DB 덮어쓰기.
4. **검증**  
   - 새 청크 길이 재분석 → `run_batch_evaluation.py`, `run_batch_rag.py`로 성능 영향 확인.

## 리스크 관리
- 긴 청크 분할 시 맥락 손실 가능성 → 요약문 삽입으로 보완.
- 자동 요약 부정확성 → 중요 본문은 요약 대신 부분 인용을 유지.
- 재임베딩 후 성능 저하 시 `evaluation_history.json`의 이전 버전과 비교해 롤백 경로 확보.

## 현재 진행 상황 요약 (2025-11-04)
- `long_chunks.jsonl`: 512 토큰 초과 청크 607건 목록
- `rechunked_long_chunks.jsonl`: 재분할된 새로운 청크 1,214건 (각 항목에 `parent_chunk_id` 포함)
- `all_chunks_rechunked.json`: 원본에서 607건 제거 후 재분할 청크 포함한 병합본 (총 2,531건)
- 다음 단계:
  1. 재분할 청크의 토큰 길이 재검증 (`analyze_chunk_lengths.py`를 재사용하거나 샘플 검토)
  2. 기존 `all_chunks.json`과 병합/교체 전략 수립 (예: parent 제거 후 새 청크 삽입) → 병합본 생성 완료
  3. `all_chunks_rechunked.json`을 기준으로 임베딩/벡터 DB 재생성, 평가 스위트로 성능 영향 확인

## 차후 절차 가이드
1. **검증**: `analyze_chunk_lengths.py --chunk-file PJ/Chat/eval/all_chunks_rechunked.json`으로 전체 갱신본 토큰 분포 확인.
2. **백업**: 기존 `2_Chunking/output/all_chunks.json`을 백업(`all_chunks_backup_YYYYMMDD.json`) 후 교체 여부 결정.
3. **임베딩 재생성**: `3_Embedding/embedder.py`를 재실행하여 새 청크 기반 임베딩 생성 (`model_name_safe` 디렉터리 신규 생성 권장).
4. **벡터 DB 갱신**: `4_Vector_DB/build_vectordb.py` 등으로 벡터 DB 재구축.
5. **평가**: `run_batch_evaluation.py`, `run_batch_rag.py` 실행 → `evaluation_history.json`, `results/` 폴더에 비교 수치 기록.
6. **회귀 대비**: 필요 시 기존 벡터 DB/임베딩 백업본을 유지해 두고, 성능 하락 시 롤백 전략 준비.

이 계획에 따라 512 토큰 초과 청크를 단계적으로 정비하면, 리랭커와 LLM 모두 입력 한계를 벗어나지 않도록 안정화할 수 있습니다.
