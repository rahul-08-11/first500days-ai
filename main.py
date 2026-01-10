import fastapi
from utils.helpers import setup_logging
setup_logging() ## set logging configuration
from dotenv import load_dotenv
load_dotenv()
from service.rag import RAGClient
from service.memory import SessionMemory
from service.azure_openai import AzureOpenAI
from pydantic import BaseModel


import logging
logger = logging.getLogger(__name__)

## FastAPI app initialization
app = fastapi.FastAPI()

## Service initializations
rag_client = RAGClient()
session_memory = SessionMemory()
azure_openai = AzureOpenAI()
logger.info("FastAPI app and services initialized.")

class AskRequest(BaseModel):
    session_id: str
    question: str

@app.post("/ask")
def ask_rag_agent(request: AskRequest):
    logger.info(f"Received /ask request: session_id={request.session_id}, question={request.question}")
    try:
        
        # Retrieve session memory
        session_history = session_memory.get_session(request.session_id)

        # Retrieve relevant documents from RAG
        logger.info(f"Searching for relevant document chunks for the question.")
        results = rag_client.search_similar_chunks(request.question, top_k=3)
        sources = set([result.get("metadata", {}).get("source", "Unknown") for result in results])


        # Generate RAG-based response
        logger.info(f"Generating RAG-based response using Azure OpenAI.")
        response = azure_openai.generate_response_v1(
            user_query=request.question,
            raw_context=results,
            memory=session_history
        )
        if response.choices[0].message.tool_calls:
            ai_response = azure_openai.process_tool_calls(request.question, response)
        else:
            ai_response = response.choices[0].message.content
        logger.info(f"AI response generated : {ai_response}")


        # Update session memory
        session_memory.add_user_message(request.session_id, request.question)
        session_memory.add_assistant_message(request.session_id, ai_response)

        return {"answer": ai_response, "source": sources}

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        return fastapi.Response(
            content="Internal server error",
            status_code=500
        )

@app.post("/v0/ask")
def ask_rag_agent(request: AskRequest):
    # Retrieve session memory
    logger.info(f"Received /v0/ask request: session_id={request.session_id}, question={request.question}")

    try:
        session_history = session_memory.get_session(request.session_id)

        # Generate response using Azure OpenAI
        logger.info(f"Generating response using Azure OpenAI v0 method.")
        ai_response = azure_openai.generate_response_v0(
            user_query=request.question,
            memory=session_history
        )
        logger.info(f"AI response generated : {ai_response}")
        # Update session memory
        session_memory.add_user_message(request.session_id, request.question)
        session_memory.add_assistant_message(request.session_id, ai_response)

        return {"answer": ai_response}
    except Exception as e:
        logger.error(f"Error in /legacy/ask endpoint: {e}")
        return fastapi.Response(
            content="Internal server error",
            status_code=500
        )
