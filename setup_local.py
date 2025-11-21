import os
import sys
import subprocess
import time
import json

PROJECT_DIR = os.path.abspath(".")
APP_DIR = os.path.join(PROJECT_DIR, "rag_app")
DOCS_DIR = os.path.join(APP_DIR, "docs")
RAG_DIR = os.path.join(APP_DIR, "rag_storage")
VENV = os.path.join(APP_DIR, "venv")
PIP = os.path.join(VENV, "Scripts", "pip.exe")
PYTHON = os.path.join(VENV, "Scripts", "python.exe")
ENV_FILE = os.path.join(APP_DIR, ".env")
RUN_FILE = os.path.join(APP_DIR, "run_rag.py")

LM_API = "http://localhost:1234/v1"
MODEL = "openai/gpt-oss-20b"
EMBED = "text-embedding-nomic-embed-text-v1.5"

REQUIRED_PACKAGES = [
    "raganything",
    "lightrag",
    "openai",
    "python-dotenv",
    "aiohttp",
    "pillow",
    "pymupdf",
    "pytesseract",
    "python-docx",
    "python-pptx",
    "pandas",
    "openpyxl",
    "xlrd",
    "aiofiles"
]

errors = []

def log(x): print(f"[INFO] {x}")
def warn(x): print(f"[WARN] {x}")
def fail(x): 
    print(f"[ERROR] {x}")
    errors.append(x)

def run(cmd):
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

# -------------------- SAFE PIP UPGRADE --------------------

def safe_pip_upgrade():
    print("[INFO] Trying safe pip upgrade (will not throw errors)...")
    try:
        result = subprocess.run(
            f'"{PIP}" install --upgrade pip',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            warn("Pip upgrade failed but ignored.")
        else:
            log("Pip upgraded successfully.")
    except Exception as e:
        warn(f"Skipping pip upgrade due to: {e}")

# -------------------- SCRIPT START --------------------

log("Creating folders...")
os.makedirs(APP_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)

# -------------------- venv --------------------

if not os.path.exists(VENV):
    log("Creating virtual environment...")
    run(f'"{sys.executable}" -m venv "{VENV}"')
else:
    log("venv already exists")

# -------------------- pip upgrade (safe) --------------------

safe_pip_upgrade()

# -------------------- package install --------------------

log("Installing required packages...")
for pkg in REQUIRED_PACKAGES:
    r = run(f'"{PIP}" install {pkg}')
    if r.returncode != 0:
        fail(f"Failed installing {pkg}: {r.stderr}")

# -------------------- .env --------------------

log("Writing .env...")
with open(ENV_FILE, "w") as f:
    f.write(f"""LLM_BINDING=lmstudio
LLM_MODEL={MODEL}
LLM_BINDING_HOST={LM_API}
LLM_BINDING_API_KEY=lm-studio

EMBEDDING_BINDING=lmstudio
EMBEDDING_MODEL={EMBED}
EMBEDDING_BINDING_HOST={LM_API}
EMBEDDING_BINDING_API_KEY=lm-studio
""")

# -------------------- run_rag.py --------------------

log("Writing run_rag.py...")
with open(RUN_FILE, "w") as f:
    f.write(r'''
import os
import asyncio
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

load_dotenv()
BASE = os.getenv("LLM_BINDING_HOST")
KEY = os.getenv("LLM_BINDING_API_KEY")
MODEL = os.getenv("LLM_MODEL")
EMBED = os.getenv("EMBEDDING_MODEL")

async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await openai_complete_if_cache(
        model=MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=BASE,
        api_key=KEY,
        max_tokens=512,
        temperature=0.3,
    )

async def embed(texts):
    client = AsyncOpenAI(base_url=BASE, api_key=KEY)
    res = await client.embeddings.create(model=EMBED, input=texts)
    return [x.embedding for x in res.data]

async def init_rag():
    cfg = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True
    )
    rag = RAGAnything(cfg, llm_model_func=llm,
                      embedding_func=EmbeddingFunc(768, 8192, embed))
    rag._mark_multimodal_processing_complete = lambda _: None
    return rag

async def process_docs(rag):
    docs = "./docs"
    done = set(os.listdir("./rag_storage/documents")) if os.path.exists("./rag_storage/documents") else set()
    for f in os.listdir(docs):
        if f in done: continue
        print("Processing", f)
        try:
            await rag.process_document_complete(
                os.path.join(docs, f),
                "./rag_storage/outputs",
                parse_method="auto",
                display_stats=True
            )
        except Exception as e:
            print("Failed", f, e)

async def main():
    rag = await init_rag()
    await process_docs(rag)
    while True:
        q = input("Ask (exit to quit): ")
        if q == "exit": break
        print(await rag.aquery(q, mode="hybrid"))

if __name__ == "__main__":
    asyncio.run(main())
''')

# -------------------- SUMMARY --------------------

print("\n===============================")
print(" SETUP COMPLETE (LOCAL ONLY)")
print("===============================")

if errors:
    print("\n⚠ The following non-critical errors occurred:")
    for e in errors:
        print(" -", e)

print("""
Next steps:

1) Install LM Studio manually (https://lmstudio.ai)
2) Inside LM Studio:
     - Search: GPT-OSS 20B
     - Download
     - Load model
     - Start SERVER (Server tab → Start Server)

3) Run:

   cd rag_app
   venv\\Scripts\\activate
   python run_rag.py

You're now fully 100% LOCAL.
""")
