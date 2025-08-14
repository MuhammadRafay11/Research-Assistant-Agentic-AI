import os
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_openai import OpenAIEmbeddings

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Connect to Pinecone index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeStore(index, embeddings.embed_query, "text")

# PDF search tool
def search_pdf(query):
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

pdf_tool = Tool(
    name="PDF Search",
    func=search_pdf,
    description="Use this to find information inside the uploaded PDF document."
)

# Wikipedia search tool
wiki_api = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia Search",
    func=wiki_api.run,
    description="Search Wikipedia for general world knowledge."
)

# Add memory so it remembers the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent with both tools
agent = initialize_agent(
    tools=[pdf_tool, wiki_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory
)

# Start interaction
print("Agent ready! Ask your question:")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.run(query)
    print("Agent:", response)
