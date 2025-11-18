import json
from pathlib import Path
from tqdm import tqdm

# RecursiveCharacterTextSplitter는 langchain_text_splitters 패키지에 있습니다.
# 먼저 해당 패키지를 설치해야 할 수 있습니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # 0. 경로 설정
    preprocessed_path = Path('/home/pencilfoxs/History_Docent_PJ_gemini/1_Data_Preprocessing/output')
    chunked_output_path = Path('/home/pencilfoxs/History_Docent_PJ_gemini/2_Chunking/output')
    chunked_output_path.mkdir(exist_ok=True)

    # 1. RecursiveCharacterTextSplitter 생성
    # 한국어 처리를 위해 separators에 문장과 문단을 구분하는 기준을 명시
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Chunk의 최대 크기 (글자 수 기준)
        chunk_overlap=100,    # Chunk 간의 중첩되는 글자 수
        separators=["\\n\\n", "\\n", ". ", "? ", "! "], # 분할 기준
        length_function=len,
    )

    all_chunks = []
    chunk_counters = {}

    # 2. 파일 처리 및 Chunking
    for text_file in tqdm(sorted(preprocessed_path.glob("*.txt")), desc="파일별 Chunking 진행"):
        series_name = text_file.stem.replace("_preprocessed", "")
        print(f"\\n--- '{series_name}' Chunking 시작 ---")

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
        print(f"총 {len(chunks)}개의 Chunk로 분할되었습니다.")

        start_index = chunk_counters.get(series_name, 0)

        for offset, chunk_text in enumerate(chunks):
            chunk_id = f"{series_name}_{start_index + offset:05d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": series_name,
                "text": chunk_text
            })

        chunk_counters[series_name] = start_index + len(chunks)

    # 3. 결과 저장
    output_filename = chunked_output_path / "all_chunks.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\\n--- Chunking 완료 ---")
    print(f"총 {len(all_chunks)}개의 Chunk가 생성되었습니다.")
    print(f"결과가 '{output_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
