from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama   
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.documents import Document
import re
from node import agent_node,qa_tool,summary_tool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Initialize Fast API application
app = FastAPI()

# Global Variable
retriever = None
llm = None

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


class QueryResponse(BaseModel):
    route: str
    result: str
    query: str

class HealthResponse(BaseModel):
    status: str
    message: str


def load_documents():
    # Load file documents
    loader_bug_report = TextLoader("ai_test_bug_report.txt")
    loader_user_feedback = TextLoader("ai_test_user_feedback.txt")

    doc_bug_report = loader_bug_report.load()
    doc_user_feedback = loader_user_feedback.load()
    
    return [doc_bug_report,doc_user_feedback]

def splitter(docs_list,file_name_list):
    parts_bug_report = re.split(r"(?=Bug\s+#\d+\b)", docs_list[0][0].page_content)
    parts_bug_report = [i.strip() for i in parts_bug_report]
    documents1 = [Document(page_content=t, metadata={"source": file_name_list[0]}) for t in parts_bug_report]

    parts_user_feedback = re.split(r"(?=Feedback\s+#\d+\b)", docs_list[1][0].page_content)
    parts_user_feedback = [i.strip() for i in parts_user_feedback]
    documents2 = [Document(page_content=t, metadata={"source": file_name_list[1]}) for t in parts_user_feedback]

    return documents1+documents2

def process_query(query: str, retriever, llm) -> dict:
    routing_decision = agent_node(query, llm)

    if routing_decision == "summary":
        result = summary_tool(query, retriever, llm)
        return {
            "route": "summary",
            "result": result
        }
    else:  # qa
        result = qa_tool(query, retriever, llm)
        return {
            "route": "qa",
            "result": result
        }

def initialize_rag_system():
    global retriever, llm
    
    # Load and process documents
    docs_list = load_documents()
    docs_splitter = splitter(docs_list, ["ai_test_bug_report.txt", "ai_test_user_feedback.txt"])

    # Create vector store and retriever
    retriever = Chroma.from_documents(
        documents=docs_splitter,
        embedding=OllamaEmbeddings(model="nomic-embed-text",base_url=OLLAMA_BASE_URL),
        collection_name="rag-chroma",
        persist_directory="./chromadb"
    ).as_retriever()

    # Initialize LLM
    llm = ChatOllama(temperature=0, model="mistral:instruct",  base_url=OLLAMA_BASE_URL)

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when API starts"""
    import time
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            initialize_rag_system()
            print("✅ RAG system initialized successfully")
            return
        except Exception as e:
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Error initializing RAG system after {max_retries} attempts: {str(e)}")
                raise

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "online",
        "message": "RAG Query API is running"
    }

@app.get("/ask")
async def ask_via_query_param(q: str):
    """Query โดยส่งผ่าน query parameter ?q="""
    if retriever == None or llm == None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        response = process_query(q, retriever, llm)
        return {
            "route": response["route"],
            "result": response["result"],
            "query": q
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)