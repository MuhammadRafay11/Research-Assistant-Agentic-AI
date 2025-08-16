import os
import uuid
import tempfile
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Fixed imports for LangChain document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Updated import for modern Pinecone SDK
from pinecone import Pinecone
import PyPDF2

# Load environment variables
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate required environment variables (removed PINECONE_ENV as it's not needed)
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, TAVILY_API_KEY]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Assistant API",
    description="Backend API for AI Research Assistant with document upload, vector search, and web search capabilities",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone with modern SDK
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    # Create vectorstore using langchain-pinecone
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings
    )
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX}")
except Exception as e:
    raise ValueError(f"Error connecting to Pinecone index '{PINECONE_INDEX}': {e}")

# Initialize Tavily Search
try:
    tavily_search = TavilySearch(api_key=TAVILY_API_KEY)
    print("✅ Tavily Search initialized")
except Exception as e:
    raise ValueError(f"Error initializing Tavily Search: {e}")

# Store uploaded file namespaces in memory (in production, use a database)
uploaded_namespaces = {}

def load_file_content(file_path: str) -> str:
    """Load content from PDF or TXT file"""
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

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Research Assistant API", 
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - POST - Upload documents",
            "query": "/query - POST - Search and get answers", 
            "health": "/health - GET - Health check",
            "namespaces": "/namespaces - GET - View uploaded files",
            "docs": "/docs - GET - API documentation"
        }
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Check file size (limit to 10MB)
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load and process the file
            text_content = load_file_content(tmp_file_path)
            
            if not text_content.strip():
                raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text_content)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No content chunks could be created from the file")
            
            # Create unique namespace
            namespace_id = str(uuid.uuid4())
            
            # Create embeddings and store in Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only process non-empty chunks
                    vector = embeddings.embed_query(chunk)
                    vectors_to_upsert.append((
                        f"{file.filename}-{namespace_id}-{i}", 
                        vector, 
                        {"text": chunk, "source": file.filename, "namespace": namespace_id}
                    ))
            
            if not vectors_to_upsert:
                raise HTTPException(status_code=400, detail="No valid content found to store")
            
            # Upsert to Pinecone with namespace
            index.upsert(vectors_to_upsert, namespace=namespace_id)
            
            # Store namespace info
            uploaded_namespaces[namespace_id] = {
                "filename": file.filename,
                "chunks_count": len(vectors_to_upsert),
                "file_size": file.size,
                "upload_time": str(uuid.uuid4())  # Simple timestamp placeholder
            }
            
            return JSONResponse(content={
                "message": f"Successfully processed and stored {len(vectors_to_upsert)} chunks",
                "namespace": namespace_id,
                "filename": file.filename,
                "chunks_count": len(vectors_to_upsert),
                "file_size": file.size,
                "status": "success"
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query")
async def query_documents(query: str = Form(...)):
    """Query the knowledge base and web"""
    try:
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long. Maximum 1000 characters allowed.")
        
        # Search results storage
        context_parts = []
        sources = []
        
        # 1. Search Pinecone vector database
        pinecone_results = None
        try:
            query_embedding = embeddings.embed_query(query)
            pinecone_results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            if pinecone_results['matches']:
                context_parts.append("**Vector Database Results:**")
                for match in pinecone_results['matches']:
                    if match.get('score', 0) > 0.7:  # Only include high-confidence matches
                        if 'metadata' in match and 'text' in match['metadata']:
                            context_parts.append(match['metadata']['text'])
                            if 'source' in match['metadata']:
                                sources.append(f"Document: {match['metadata']['source']}")
                
                if any("Document:" in s for s in sources):
                    sources.append("Vector Database")
                        
        except Exception as e:
            print(f"Pinecone search error: {e}")
        
        # 2. Search the web using Tavily
        web_results = None
        try:
            web_results = tavily_search.invoke({"query": query, "max_results": 3})
            
            if web_results and "results" in web_results and web_results["results"]:
                context_parts.append("\n**Web Search Results:**")
                for result in web_results["results"]:
                    if result.get("content") and len(result["content"]) > 50:  # Filter out very short results
                        context_parts.append(result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"])
                        if result.get("url"):
                            sources.append(f"Web: {result.get('title', 'Web Result')} ({result['url']})")
                
                if any("Web:" in s for s in sources):
                    sources.append("Web Search (Tavily)")
                        
        except Exception as e:
            print(f"Web search error: {e}")
        
        # Prepare context for LLM
        if not context_parts:
            context = "No specific context found in uploaded documents or web search. Please provide a general answer based on your knowledge."
            sources = ["General Knowledge"]
        else:
            context = "\n".join(context_parts)
        
        # Generate answer using LLM
        final_prompt = f"""
        Answer the following question based on the provided context. If the context doesn't contain 
        relevant information, provide a helpful answer based on your general knowledge.
        Be comprehensive but concise, and clearly indicate when you're drawing from the provided context 
        versus general knowledge.
        
        Context:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive and helpful answer:
        """
        
        try:
            answer = llm.predict(final_prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
        
        # Remove duplicates from sources while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)
        
        return JSONResponse(content={
            "answer": answer,
            "sources": unique_sources,
            "query": query,
            "has_vector_results": pinecone_results is not None and len(pinecone_results.get('matches', [])) > 0,
            "has_web_results": web_results is not None and len(web_results.get("results", [])) > 0,
            "status": "success"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Pinecone connection
        index_stats = index.describe_index_stats()
        pinecone_status = "healthy"
    except:
        pinecone_status = "unhealthy"
    
    try:
        # Test OpenAI connection with a simple embedding
        embeddings.embed_query("test")
        openai_status = "healthy"
    except:
        openai_status = "unhealthy"
    
    return {
        "status": "healthy",
        "message": "AI Research Assistant API is running",
        "components": {
            "pinecone": pinecone_status,
            "openai": openai_status,
            "tavily": "configured"
        },
        "uploaded_files": len(uploaded_namespaces)
    }

@app.get("/namespaces")
async def get_namespaces():
    """Get information about uploaded file namespaces"""
    return {
        "total_files": len(uploaded_namespaces),
        "namespaces": uploaded_namespaces
    }

@app.delete("/namespace/{namespace_id}")
async def delete_namespace(namespace_id: str):
    """Delete a specific namespace and its vectors"""
    try:
        if namespace_id not in uploaded_namespaces:
            raise HTTPException(status_code=404, detail="Namespace not found")
        
        # Delete from Pinecone
        index.delete(delete_all=True, namespace=namespace_id)
        
        # Remove from local storage
        filename = uploaded_namespaces[namespace_id]["filename"]
        del uploaded_namespaces[namespace_id]
        
        return {
            "message": f"Successfully deleted namespace {namespace_id} and file {filename}",
            "deleted_namespace": namespace_id,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting namespace: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(" Starting AI Research Assistant API...")
    print(" API Documentation: http://localhost:8000/docs")
    print(" Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)