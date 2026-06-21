from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files are handled natively by Vercel.

class ChatRequest(BaseModel):
    document_text: str
    query: str
    chat_history: list

@app.post("/api/upload")
async def upload_document(file: List[UploadFile] = File(...)):
    try:
        combined_text = ""
        for f in file:
            file_extension = os.path.splitext(f.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await f.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            if file_extension == '.pdf':
                loader = PyMuPDFLoader(temp_file_path)
                documents = loader.load()
            elif file_extension == '.txt':
                loader = TextLoader(temp_file_path, encoding='utf-8')
                documents = loader.load()
            else:
                return JSONResponse(status_code=400, content={"error": f"Unsupported file format: {f.filename}"})

            file_text = "\n".join([doc.page_content for doc in documents])
            combined_text += f"\n--- Start of {f.filename} ---\n{file_text}\n--- End of {f.filename} ---\n"
            os.remove(temp_file_path)

        # For stateless, we return the text to the client
        return {"text": combined_text[:200000]} # Increased to 200k chars for larger documents
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful research assistant. Answer the user's questions based on the following document text. If the answer is not in the text, say you don't know.\n\nDocument Text:\n{document_text}"),
            ("placeholder", "{chat_history}"),
            ("user", "{query}")
        ])
        
        chain = prompt | llm
        
        # Format chat history for LangChain
        formatted_history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                formatted_history.append(("user", msg["content"]))
            else:
                formatted_history.append(("assistant", msg["content"]))

        response = chain.invoke({
            "document_text": request.document_text,
            "chat_history": formatted_history,
            "query": request.query
        })
        
        return {"answer": response.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/presentation")
async def generate_presentation(request: ChatRequest):
    try:
        llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert presentation generator. Create a structured slide-by-slide presentation summary based on the provided documents. Use clear slide numbers, titles, and bullet points. Make it well formatted.\n\nDocuments:\n{document_text}"),
            ("user", "Please generate a presentation based on the uploaded files.")
        ])
        
        chain = prompt | llm
        
        response = chain.invoke({
            "document_text": request.document_text
        })
        
        return {"presentation": response.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
