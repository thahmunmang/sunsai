import os
import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

app = modal.App("sunsai-gemma4-vllm")

# Define container image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.44.0",
    "fastapi[standard]",
    "pydantic>=2.0"
)

web_app = FastAPI(title="Sunsai AI Studio Modal Backend")

# ==========================================
# CRITICAL: ENABLE CORS FOR GITHUB PAGES
# ==========================================
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from your GitHub Pages domain and local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    messages: List[Dict[str, str]]
    language: Optional[str] = "hakha"
    model_tier: Optional[str] = "Sunsai Auto"

@web_app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    # Your model generation logic here (vllm or transformers inference)
    # Example response structure:
    return JSONResponse(content={
        "reply": "Biahalnak hi ka hngalhthiam. Sunsai AI on Modal is connected successfully!",
        "mode": "chat"
    })

@app.function(image=image, gpu="A10G", timeout=600)
@modal.asgi_app()
def serve():
    return web_app
