from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import get_response

app = FastAPI(title="AI University Chatbot API")

# ✅ CORS configuration
origins = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # use ["*"] for quick testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request / Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    reply: str

# ✅ API endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = get_response(req.question)
    return {"reply": reply}