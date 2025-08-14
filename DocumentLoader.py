import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import PyPDF2

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# 1. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Function to load text from TXT or PDF
def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    else:
        raise ValueError("Unsupported file type. Please use .txt or .pdf")

# 2. Choose your file
file_path = r"F:\Research Assistance (Agentic AI)\Science-ML-2015.pdf"  
text = load_file(file_path)

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# 4. Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 5. Upload chunks to Pinecone
for i, chunk in enumerate(chunks):
    vector = embeddings.embed_query(chunk)
    index.upsert([(f"{os.path.basename(file_path)}-{i}", vector, {"text": chunk})])

print(f"Uploaded {len(chunks)} chunks from '{file_path}' to Pinecone index '{PINECONE_INDEX}'")
