import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load .env file
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_REGION = os.getenv("PINECONE_ENV")  # e.g., us-east-1-aws

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,  # match OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # from your PINECONE_ENV
    )
    print(f"Index '{PINECONE_INDEX}' created.")
else:
    print(f"Index '{PINECONE_INDEX}' already exists.")

# Connect to the index
index = pc.Index(PINECONE_INDEX)
print(f"Connected to index: {index}")
