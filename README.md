# AI Research Assistant - Agentic System

## AI Research Assistant with Multi-Source Intelligence & Agentic Decision-Making

Developed a sophisticated AI research assistant that goes beyond traditional RAG (Retrieval Augmented Generation) by implementing true agentic behavior with autonomous decision-making capabilities. The system intelligently orchestrates multiple knowledge sources including user-uploaded documents (PDF/TXT), real-time web search, and Wikipedia to provide comprehensive, contextually-aware responses.

## Key Technical Achievements:

Built FastAPI-based backend with RESTful endpoints for document upload, processing, and intelligent querying
Implemented vector similarity search using Pinecone vector database with OpenAI embeddings (text-embedding-3-small)
Integrated multi-source data retrieval: document vectorstore, Tavily web search API, and Wikipedia API
Designed namespace-based document isolation for multi-user/multi-document scenarios
Developed hybrid context synthesis algorithm that dynamically weighs and merges information from different sources
Implemented document chunking and preprocessing pipeline for efficient vector storage and retrieval

### Technologies
Python, FastAPI, OpenAI GPT-4, Pinecone Vector Database, LangChain, Tavily API, PyPDF2, CORS middleware

### Main Files
FastAPIVersion.py - Frontend.html
rest are for testing purpose

### Instructions
run the FastAPI server using  "uvicorn FastAPIVersion:app --reload"
or if you want to run it on Different port "uvicorn FastAPIVersion:app --reload --port=8000" update the number accordingly

make sure to have well structured .env file with your API's

Last make sure to install the dependencies using requirements.txt