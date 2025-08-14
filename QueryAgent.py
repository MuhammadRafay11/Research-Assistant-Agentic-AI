import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# 3. Initialize embeddings + OpenAI client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# 4. Function to query Pinecone and get answer
def query_pdf(question):
    # Convert question into embedding
    query_embedding = embeddings.embed_query(question)

    # Search in Pinecone
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Extract text chunks
    context = "\n".join([match['metadata']['text'] for match in search_results['matches']])

    # Send to OpenAI for answering
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question about your PDF (or type 'exit'): ")
        if question.lower() == "exit":
            break
        answer = query_pdf(question)
        print("\nAnswer:", answer)
