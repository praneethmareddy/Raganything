@echo off
echo ==========================================
echo RAG-ANYTHING + LM STUDIO WINDOWS SETUP
echo ==========================================

REM --- Create directories ---
mkdir rag_app
cd rag_app
mkdir docs
mkdir rag_storage

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing core dependencies...
pip install raganything lightrag openai python-dotenv aiohttp

echo Installing document processing packages...
pip install python-docx python-pptx pandas openpyxl xlrd
pip install pillow pymupdf pytesseract aiofiles

echo Installing optional OCR engine...
echo If Tesseract is not installed, download from:
echo https://github.com/UB-Mannheim/tesseract/wiki
echo (Skip if already installed)
echo.

REM --- Create .env file ---
echo Creating .env ...
(
echo LLM_BINDING=lmstudio
echo LLM_MODEL=openai/gpt-oss-20b
echo LLM_BINDING_HOST=http://localhost:1234/v1
echo LLM_BINDING_API_KEY=lm-studio
echo.
echo EMBEDDING_BINDING=lmstudio
echo EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
echo EMBEDDING_BINDING_HOST=http://localhost:1234/v1
echo EMBEDDING_BINDING_API_KEY=lm-studio
) > .env

echo Creating run_rag.py ...

REM --- Write Python script ---

(
echo import os
echo import asyncio
echo import uuid
echo from dotenv import load_dotenv
echo.
echo from raganything import RAGAnything, RAGAnythingConfig
echo from lightrag.llm.openai import openai_complete_if_cache
echo from lightrag.utils import EmbeddingFunc
echo from openai import AsyncOpenAI
echo.
echo load_dotenv()
echo.
echo LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost^:1234/v1")
echo LM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
echo LM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
echo.
echo EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
echo.
echo.
echo async def lmstudio_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
echo ^    return await openai_complete_if_cache(
echo ^        model=LM_MODEL,
echo ^        prompt=prompt,
echo ^        system_prompt=system_prompt,
echo ^        history_messages=history_messages or [],
echo ^        base_url=LM_BASE_URL,
echo ^        api_key=LM_API_KEY,
echo ^        max_tokens=512,
echo ^        temperature=0.3,
echo ^    )
echo.
echo.
echo async def lmstudio_embedding(texts):
echo ^    client = AsyncOpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
echo ^    result = await client.embeddings.create(
echo ^        model=EMBED_MODEL,
echo ^        input=texts
echo ^    )
echo ^    return [e.embedding for e in result.data]
echo.
echo.
echo async def init_rag():
echo ^    config = RAGAnythingConfig(
echo ^        working_dir="./rag_storage",
echo ^        parser="mineru",
echo ^        parse_method="auto",
echo ^        enable_image_processing=True,
echo ^        enable_table_processing=True,
echo ^        enable_equation_processing=True,
echo ^    )
echo.
echo ^    rag = RAGAnything(
echo ^        config=config,
echo ^        llm_model_func=lmstudio_llm,
echo ^        embedding_func=EmbeddingFunc(
echo ^            embedding_dim=768,
echo ^            max_token_size=8192,
echo ^            func=lmstudio_embedding,
echo ^        ),
echo ^    )
echo.
echo ^    async def _noop(doc_id: str):
echo ^        return None
echo.
echo ^    rag._mark_multimodal_processing_complete = _noop
echo.
echo ^    return rag
echo.
echo.
echo async def process_all_docs(rag):
echo ^    docs_folder = "./docs"
echo ^    os.makedirs(docs_folder, exist_ok=True)
echo.
echo ^    processed_folder = "./rag_storage/documents"
echo ^    existing = set(os.listdir(processed_folder)) if os.path.exists(processed_folder) else set()
echo.
echo ^    for file in os.listdir(docs_folder):
echo ^        path = os.path.join(docs_folder, file)
echo.
echo ^        if file in existing:
echo ^            print(f"🔵 Skipping already processed: {file}")
echo ^            continue
echo.
echo ^        print(f"🟢 Processing new file: {file}")
echo ^        try:
echo ^            await rag.process_document_complete(
echo ^                file_path=path,
echo ^                output_dir="./rag_storage/outputs",
echo ^                parse_method="auto",
echo ^                display_stats=True,
echo ^            )
echo ^            print(f"✅ Done: {file}")
echo ^        except Exception as e:
echo ^            print(f"❌ Failed {file}: {e}")
echo.
echo.
echo async def query_loop(rag):
echo ^    while True:
echo ^        user_input = input("\n❓ Ask a question (or type 'exit'): ")
echo ^        if user_input.lower() in ["exit", "quit"]:
echo ^            break
echo.
echo ^        answer = await rag.aquery(user_input, mode="hybrid")
echo ^        print(f"\n💬 Answer:\n{answer}\n")
echo.
echo.
echo async def main():
echo ^    print("🔧 Initializing RAG-Anything with LM Studio...")
echo ^    rag = await init_rag()
echo.
echo ^    print("📄 Scanning docs/ for new files...")
echo ^    await process_all_docs(rag)
echo.
echo ^    print("💬 Entering query mode...")
echo ^    await query_loop(rag)
echo.
echo.
echo if __name__ == "__main__":
echo ^    asyncio.run(main())
) > run_rag.py

echo ==========================================
echo SETUP COMPLETE!
echo ==========================================

echo To start the app:
echo 1. Ensure LM Studio server is running on port 1234
echo 2. Activate venv: venv\Scripts\activate
echo 3. Run: python run_rag.py

pause
