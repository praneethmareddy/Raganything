import os
import subprocess
import sys

print("=== RAG-ANYTHING + LM STUDIO SETUP FOR WINDOWS ===")

# Create folder structure
os.makedirs("rag_app/docs", exist_ok=True)
os.makedirs("rag_app/rag_storage", exist_ok=True)

os.chdir("rag_app")

print("Creating virtual environment...")
subprocess.run([sys.executable, "-m", "venv", "venv"])

print("Installing packages...")
subprocess.run(["venv\\Scripts\\pip.exe", "install", "--upgrade", "pip"])
subprocess.run(["venv\\Scripts\\pip.exe", "install",
                "raganything", "lightrag", "openai", "python-dotenv",
                "aiohttp", "pillow", "pymupdf", "pytesseract",
                "python-docx", "python-pptx", "pandas", "openpyxl", "xlrd"])

print("Creating .env file...")
with open(".env", "w") as f:
    f.write("""LLM_BINDING=lmstudio
LLM_MODEL=openai/gpt-oss-20b
LLM_BINDING_HOST=http://localhost:1234/v1
LLM_BINDING_API_KEY=lm-studio

EMBEDDING_BINDING=lmstudio
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDING_BINDING_HOST=http://localhost:1234/v1
EMBEDDING_BINDING_API_KEY=lm-studio
""")

print("Creating run_rag.py...")
with open("run_rag.py", "w") as f:
    f.write(r'''
import os
import asyncio
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

load_dotenv()

LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
LM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

async def lmstudio_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await openai_complete_if_cache(
        model=LM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
        max_tokens=512,
        temperature=0.3,
    )

async def lmstudio_embedding(texts):
    client = AsyncOpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
    result = await client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [e.embedding for e in result.data]

async def init_rag():
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=lmstudio_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lmstudio_embedding,
        ),
    )

    async def _noop(doc_id: str):
        return None

    rag._mark_multimodal_processing_complete = _noop
    return rag

async def process_all_docs(rag):
    docs_folder = "./docs"
    os.makedirs(docs_folder, exist_ok=True)

    processed_folder = "./rag_storage/documents"
    existing = set(os.listdir(processed_folder)) if os.path.exists(processed_folder) else set()

    for file in os.listdir(docs_folder):
        path = os.path.join(docs_folder, file)

        if file in existing:
            print(f"Skipping already processed: {file}")
            continue

        print(f"Processing new file: {file}")
        try:
            await rag.process_document_complete(
                file_path=path,
                output_dir="./rag_storage/outputs",
                parse_method="auto",
                display_stats=True,
            )
            print(f"Done: {file}")
        except Exception as e:
            print(f"Failed {file}: {e}")

async def query_loop(rag):
    while True:
        user_input = input("\nAsk a question ('exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        answer = await rag.aquery(user_input, mode="hybrid")
        print(f"\nAnswer:\n{answer}\n")

async def main():
    print("Initializing RAG with LM Studio...")
    rag = await init_rag()

    print("Scanning docs/ for new files...")
    await process_all_docs(rag)

    print("Entering query mode...")
    await query_loop(rag)

if __name__ == "__main__":
    asyncio.run(main())
''')

print("🎉 Setup Complete!")
print("\nTo run:")
print("cd rag_app")
print("venv\\Scripts\\activate")
print("python run_rag.py")
