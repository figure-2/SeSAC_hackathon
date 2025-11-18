# 역사 도슨트 RAG 파이프라인 v1.0

## 1. 프로젝트 개요
이 프로젝트는 한국사 원문 데이터를 기반으로 사용자의 질문에 답변하는 RAG(검색 증강 생성) 시스템의 전체 구축 과정을 담고 있습니다. 데이터 전처리부터 모델 파인튜닝, 벡터 DB 구축, 그리고 LLM을 이용한 답변 생성 및 평가까지의 전 과정을 재현할 수 있도록 아카이빙되었습니다.

## 2. 환경 설정
```bash
# 1. Conda 가상환경 생성 및 활성화
conda create -n history_rag_prod python=3.10 -y
conda activate history_rag_prod

# 2. 필수 라이브러리 설치
pip install -r requirements.txt

# 3. API 키 설정
# config/api_keys.env 파일에 자신의 Google AI Studio API 키를 입력하세요.
# 예시: GOOGLE_API_KEY="여기에_API_키_입력"
```

## 3. 파이프라인 단계별 실행 가이드
모든 스크립트는 프로젝트 루트 디렉토리(`/home/pencilfoxs/History_Docent_PJ_1105`)에서 실행하는 것을 기준으로 합니다.

### Step 1: 벡터 DB 구축
(청킹, 임베딩, DB 생성을 하나의 과정으로 가정)
```bash
# 1. (선택) 청킹 재실행 (이미 결과물이 있으므로 필요 시에만)
# python 1_chunking/chunker.py

# 2. 청크 임베딩 생성
python 3_embedding/embedder.py

# 3. ChromaDB에 임베딩 저장
python 4_vector_db/build_vectordb.py
```

### Step 2: RAG 파이프라인 성능 평가
```bash
# 1. 정량 평가 (Retrieval/Rerank 성능 측정)
python 5_evaluation/run_batch_evaluation.py \
    --config config/config.yaml \
    --eval-dataset 5_evaluation/assets/eval_qa_dataset.json \
    --retrieve-k 20 \
    --rerank-k 7 \
    --top-k 7 \
    --run-name "final_quantitative_eval"

# 2. 정성 평가 (Gemini 기반 최종 답변 생성)
python 5_evaluation/run_batch_rag.py \
    --config config/config.yaml \
    --queries 5_evaluation/assets/queries.yaml \
    --retrieve-k 20 \
    --rerank-k 7 \
    --run-name "final_qualitative_eval_gemini"
```
*   평가 결과는 `5_evaluation/results/` 폴더 하위에 타임스탬프와 함께 저장됩니다.

## 4. 설정 변경
-   모델 경로, DB 경로 등 주요 설정은 `config/config.yaml` 파일에서 수정할 수 있습니다.
-   API 키는 `config/api_keys.env` 파일에서 관리됩니다.

## 5. 프로젝트 구조
- `0_data/`: 원본 데이터
- `1_chunking/`: 데이터 청킹 스크립트 및 결과
- `2_finetuning/`: 임베딩/리랭커 모델 파인튜닝 관련 (현재는 결과 모델만 포함)
- `3_embedding/`: 텍스트 임베딩 생성 스크립트
- `4_vector_db/`: 벡터 DB 구축 스크립트 및 DB 파일
- `5_evaluation/`: 정량/정성 평가 스크립트, 데이터셋, 결과
- `6_reports/`: 분석 보고서
- `utils/`: 공통 유틸리티 모듈
- `config/`: 프로젝트 설정 파일
