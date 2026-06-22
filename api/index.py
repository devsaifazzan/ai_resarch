from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import json
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
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
    api_key: str

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
        if not request.api_key:
            return JSONResponse(status_code=400, content={"error": "Gemini API Key is missing. Please provide it in the UI."})
            
        llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.5-flash", google_api_key=request.api_key)
        
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
        
        answer_content = response.content
        if isinstance(answer_content, list):
            answer_content = "\n".join([part.get("text", "") for part in answer_content if isinstance(part, dict) and "text" in part])
        elif not isinstance(answer_content, str):
            answer_content = str(answer_content)

        return {"answer": answer_content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/presentation")
async def generate_presentation(request: ChatRequest):
    try:
        if not request.api_key:
            return JSONResponse(status_code=400, content={"error": "Gemini API Key is missing. Please provide it in the UI."})
            
        llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-3.5-flash", google_api_key=request.api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert presentation generator. Create a structured slide-by-slide presentation summary based on the provided documents.\n"
                       "Return ONLY a valid, raw JSON object with NO markdown formatting, NO backticks, and NO additional text.\n"
                       "The JSON must have a single key 'slides', which is an array of objects.\n"
                       "Each slide object must have: 'title' (string), 'subtitle' (string, optional), and 'bullets' (array of strings).\n"
                       "Keep the presentation professional, concise, and structured (around 5 to 10 slides).\n"
                       "Document Text:\n{document_text}"),
            ("user", "Extract the presentation JSON.")
        ])
        
        chain = prompt | llm
        
        response = chain.invoke({
            "document_text": request.document_text
        })
        
        presentation_content = response.content
        if isinstance(presentation_content, list):
            presentation_content = "".join([part.get("text", "") for part in presentation_content if isinstance(part, dict) and "text" in part])
        elif not isinstance(presentation_content, str):
            presentation_content = str(presentation_content)
            
        presentation_content = presentation_content.strip()
        if presentation_content.startswith("```json"):
            presentation_content = presentation_content[7:]
        elif presentation_content.startswith("```"):
            presentation_content = presentation_content[3:]
        if presentation_content.endswith("```"):
            presentation_content = presentation_content[:-3]
            
        try:
            presentation_json = json.loads(presentation_content.strip())
            return {"presentation": presentation_json}
        except json.JSONDecodeError:
            return JSONResponse(status_code=500, content={"error": "Failed to parse presentation JSON from AI response."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/graph")
async def generate_graph(request: ChatRequest):
    try:
        import json
        if not request.api_key:
            return JSONResponse(status_code=400, content={"error": "Gemini API Key is missing. Please provide it in the UI."})
            
        llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-3.5-flash", google_api_key=request.api_key)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert knowledge graph extractor. Analyze the document text and extract the key entities and their relationships.\n"
                       "Return ONLY a valid, raw JSON object with NO markdown formatting, NO backticks, and NO additional text.\n"
                       "The JSON must have two arrays: 'nodes' and 'edges'.\n"
                       "- 'nodes' elements must have: 'id' (unique string), 'label' (short display name), 'group' (category string like 'Concept', 'Person', 'Organization', etc.).\n"
                       "- 'edges' elements must have: 'from' (source node id), 'to' (target node id), 'label' (relationship description).\n"
                       "Extract the most important 15-30 nodes that summarize the core knowledge.\n"
                       "Document Text:\n{document_text}"),
            ("user", "Extract the knowledge graph JSON.")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"document_text": request.document_text})
        
        graph_content = response.content
        if isinstance(graph_content, list):
            graph_content = "".join([part.get("text", "") for part in graph_content if isinstance(part, dict) and "text" in part])
        elif not isinstance(graph_content, str):
            graph_content = str(graph_content)
            
        # Clean up any potential markdown backticks just in case
        graph_content = graph_content.strip()
        if graph_content.startswith("```json"):
            graph_content = graph_content[7:]
        elif graph_content.startswith("```"):
            graph_content = graph_content[3:]
        if graph_content.endswith("```"):
            graph_content = graph_content[:-3]
            
        graph_json = json.loads(graph_content.strip())
        return graph_json
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"error": "Failed to parse graph JSON from AI response."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
