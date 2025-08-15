import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

# ---------- Load ENV ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "agenticai")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY / PINECONE_API_KEY / TAVILY_API_KEY in .env")

# ---------- LLM & Embeddings ----------
# Use 1536-dim to match index (text-embedding-3-small)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

# ---------- Pinecone ----------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist (AWS us-east-1)
existing = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX not in existing:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,              # must match embeddings model above
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Vector store wrapper (no Index object needed)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
)

# ---------- External tools ----------
wiki = WikipediaAPIWrapper()
tavily = TavilySearch(api_key=TAVILY_API_KEY)

def pinecone_context(query: str, k: int = 5) -> str:
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs) if docs else ""
    except Exception as e:
        return f"(Pinecone error: {e})"

def wikipedia_context(query: str) -> str:
    try:
        res = wiki.run(query)
        return res or ""
    except Exception as e:
        return f"(Wikipedia error: {e})"

def web_context(query: str, k: int = 5) -> str:
    try:
        # TavilySearch.run returns a concise synthesis of top results
        return tavily.run(query, max_results=k)  # safe default
    except Exception as e:
        return f"(Web search error: {e})"

def hybrid_answer(query: str) -> str:
    pc_ctx = pinecone_context(query, k=5)
    wk_ctx = wikipedia_context(query)
    wb_ctx = web_context(query, k=5)

    context = (
        f"--- [PINECONE] ---\n{pc_ctx or '(no matches)'}\n\n"
        f"--- [WIKIPEDIA] ---\n{wk_ctx or '(no results)'}\n\n"
        f"--- [WEB] ---\n{wb_ctx or '(no results)'}"
    )

    prompt = f"""
You are a research assistant. Merge the following sources into one precise answer.
- Use facts from [PINECONE] (user's PDFs), [WIKIPEDIA], and [WEB].
- Prefer [PINECONE] when it directly answers the question.
- If a source is empty, ignore it.
- Cite inline with [PINECONE], [WIKIPEDIA], or [WEB] when you use them.
- If uncertain, say what is unknown.

Question: {query}

Context:
{context}

Now write the final answer:
"""
    resp = llm.invoke(prompt)
    return resp.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Exiting agent. Goodbye!")
            break
        try:
            ans = hybrid_answer(q)
            print("\n Answer:\n", ans)
        except Exception as e:
            print(f" Error: {e}")
