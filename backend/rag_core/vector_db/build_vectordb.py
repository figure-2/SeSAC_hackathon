import argparse
import json
import shutil
from pathlib import Path

import chromadb
import torch
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings

def main(args):
    # 0. 경로 및 모델 설정
    model_path = args.model_path
    collection_name = args.collection_name
    db_path = Path(f'/home/pencilfoxs/History_Docent_PJ_gemini/4_Vector_DB/db/{collection_name}')
    if db_path.exists():
        print(f"기존 DB 경로가 발견되어 초기화합니다: {db_path}")
        shutil.rmtree(db_path)
    db_path.mkdir(exist_ok=True, parents=True)
    chunk_file_path = Path(__file__).parent.parent / "2_Chunking/output/all_chunks.json"

    print(f"--- 벡터 DB 구축 시작: {collection_name} ---")

    # 1. HuggingFace 임베딩 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"임베딩 모델 로드: {model_path} (device={device})")
    # 파인튜닝된 로컬 모델 디렉토리를 직접 로드
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. 문서(Chunk) 데이터 로드
    with open(chunk_file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"총 {len(chunks)}개의 청크 로드 완료.")

    # 3. ChromaDB 클라이언트 및 컬렉션 준비
    client = chromadb.PersistentClient(path=str(db_path))
    
    # 컬렉션 생성 시, 임베딩 모델을 embedding_function으로 지정
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None, # 직접 임베딩을 제공하므로 None으로 설정
        metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB 컬렉션 '{collection_name}' 준비 완료.")
    
    # 4. 데이터 준비 (문서, 메타데이터, ID)
    documents = [item['text'] for item in chunks]
    metadatas = [{'source': item['source']} for item in chunks]
    ids = [item['chunk_id'] for item in chunks]

    # 5. 배치 단위로 문서 임베딩 및 DB 추가
    batch_size = 32 # 임베딩 생성 및 DB 추가를 위한 배치 크기
    
    for i in tqdm(range(0, len(documents), batch_size), desc="임베딩 생성 및 DB 저장 중"):
        batch_documents = documents[i:i+batch_size]
        
        # 실시간으로 임베딩 생성
        batch_embeddings = embeddings_model.embed_documents(batch_documents)
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print(f"\n--- 벡터 DB 구축 완료: {collection_name} ---")
    print(f"총 {collection.count()}개의 아이템이 '{collection_name}' 컬렉션에 저장되었습니다.")
    print(f"DB 저장 경로: {db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 청크를 임베딩하여 ChromaDB에 저장합니다.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='HuggingFace 모델 디렉토리의 전체 경로'
    )
    parser.add_argument(
        '--collection_name',
        type=str,
        required=True,
        help='ChromaDB에 저장될 컬렉션 이름 (e.g., finetuned_ko-sroberta)'
    )
    
    args = parser.parse_args()
    main(args)
