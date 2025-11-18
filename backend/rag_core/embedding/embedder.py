import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings

def create_batches(items, batch_size):
    """리스트를 배치 크기로 나눕니다."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def main(args):
    # 0. 경로 설정
    chunk_file = Path('/home/pencilfoxs/History_Docent_PJ_gemini/2_Chunking/output/all_chunks.json')
    output_dir = Path(f'/home/pencilfoxs/History_Docent_PJ_gemini/3_Embedding/output/{args.model_name_safe}')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "embeddings.pkl"

    print(f"--- 임베딩 생성 시작: {args.model_name} ---")

    # 1. 청크 데이터 로드
    print(f"청크 파일 로드: {chunk_file}")
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"총 {len(chunks)}개의 청크 로드 완료.")

    # 2. 임베딩 모델 설정 및 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    print(f"임베딩 모델 '{args.model_name}'을 로드합니다... (device={device})")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=args.model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("모델 로드 완료.")

    # 3. 배치 처리 및 임베딩 생성
    batches = create_batches(chunks, args.batch_size)
    all_embeddings = []
    
    print(f"배치 크기: {args.batch_size}, 총 배치 수: {len(batches)}")
    for batch in tqdm(batches, desc=f"임베딩 생성 ({args.model_name_safe})"):
        texts = [item['text'] for item in batch]
        batch_embeddings = embeddings_model.embed_documents(texts)
        all_embeddings.extend(batch_embeddings)

    # 4. 결과 저장
    # numpy 배열로 변환하여 저장
    final_embeddings = np.array(all_embeddings)
    
    # 메타데이터와 임베딩을 함께 저장
    embedding_data = {
        "chunks": chunks,
        "embeddings": final_embeddings,
        "model_name": args.model_name,
        "model_name_safe": args.model_name_safe
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(embedding_data, f)

    print(f"\n--- 임베딩 완료: {args.model_name} ---")
    print(f"총 {len(final_embeddings)}개의 임베딩이 생성되었습니다.")
    print(f"결과가 '{output_file}'에 저장되었습니다.")
    print(f"임베딩 차원: {final_embeddings.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="텍스트 청크에 대한 임베딩을 생성합니다.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        help='Hugging Face의 임베딩 모델 이름'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='임베딩 처리 시의 배치 크기'
    )
    
    args = parser.parse_args()
    
    # 모델 이름을 파일 경로로 사용하기 안전하게 변경
    args.model_name_safe = args.model_name.replace('/', '_')
    
    main(args)
