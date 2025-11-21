import os
import uuid
import asyncio
from typing import List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from raganything import RAGAnything, RAGAnythingConfig

# Load environment
load_dotenv()

LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
LM_API_KEY  = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
LM_MODEL    = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")


# ---------------- LM Studio - LLM ----------------
async def lmstudio_llm(prompt: str, system_prompt: Optional[str] = None):
    """Generate text using LM Studio."""
    client = AsyncOpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = await client.chat.completions.create(
        model=LM_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# --------------- LM Studio - Embeddings ---------------
async def lmstudio_embeddings(texts: List[str]):
    client = AsyncOpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# --------------- RAG Anything Integration ---------------
class LMStudioRAG:
    def __init__(self):
        print("🚀 Initializing RAG-Anything + LM Studio...\n")

        self.config = RAGAnythingConfig(
            working_dir=f"./rag_storage_full/{uuid.uuid4()}",
            parser="mineru",
            parse_method="auto",

            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        print(f"📁 Working directory: {self.config.working_dir}")

        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=lmstudio_llm,
            embedding_func=lmstudio_embeddings,
        )

    async def process_folder(self, folder_path="docs"):
        print(f"\n📂 Scanning folder: {folder_path}")

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            if os.path.isfile(fpath):
                print(f"\n📄 Processing: {fname}")
                try:
                    await self.rag.process_document_complete(
                        file_path=fpath,
                        output_dir="./output_full",
                        parse_method="auto",
                        display_stats=False,
                    )
                    print(f"✅ Processed: {fname}")
                except Exception as e:
                    print(f"❌ Error processing {fname}: {e}")

        print("\n📚 All documents processed successfully!\n")

    async def ask(self, query: str):
        result = await self.rag.aquery(query, mode="hybrid")
        return result


# --------------- MAIN LOOP ---------------
async def main():
    rag = LMStudioRAG()

    # 1️⃣ Process docs folder
    await rag.process_folder("docs")

    print("\n====================================")
    print("  Welcome to Local LM Studio RAG")
    print("  Type your questions anytime.")
    print("  Type 'exit' to quit.")
    print("====================================\n")

    # 2️⃣ Continuous user query loop
    while True:
        user_q = input("\n❓ You: ").strip()
        if user_q.lower() in ["exit", "quit", "bye"]:
            print("\n👋 Exiting. Goodbye!\n")
            break

        print("\n⏳ Thinking...")
        answer = await rag.ask(user_q)
        print("\n🟩 Answer:\n")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
