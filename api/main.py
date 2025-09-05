#!/usr/bin/env python3
"""
Moses Omondi's AI Recruitment Assistant - Production API
========================================================

FastAPI deployment for Moses's comprehensive DevSecOps + MLOps + MLSecOps AI assistant.
Ready for website integration and production use.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
import asyncio
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Moses Omondi's AI Recruitment Assistant",
    description="Comprehensive DevSecOps + MLOps + MLSecOps AI assistant for recruitment conversations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
tokenizer = None
model_loaded = False

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    max_tokens: Optional[int] = 400
    temperature: Optional[float] = 0.7
    context: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    timestamp: str
    model_info: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    gpu_available: bool
    timestamp: str
    version: str

async def load_model():
    """Load Moses's AI recruitment assistant model."""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return
    
    try:
        logger.info("Loading Moses's AI recruitment assistant...")
        
        MODEL_PATH = "models/moses-recruitment-assistant"
        BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # 4-bit quantization for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True
        )
        
        # Load fine-tuned adapter
        logger.info("Loading fine-tuned adapter...")
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # Set to evaluation mode
        model.eval()
        
        model_loaded = True
        logger.info("✅ Moses's AI recruitment assistant loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("🚀 Starting Moses Omondi's AI Recruitment Assistant API")
    await load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Moses Omondi's AI Recruitment Assistant API",
        "description": "DevSecOps + MLOps + MLSecOps expertise for recruitment conversations",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        model_loaded=model_loaded,
        gpu_available=torch.cuda.is_available(),
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_moses(request: ChatRequest):
    """
    Chat with Moses's AI recruitment assistant.
    
    Supports recruitment conversations about:
    - DevSecOps expertise and leadership
    - MLOps and AI operations at scale
    - MLSecOps and AI security
    - Technical background and skills
    - Executive-level AI and security roles
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait and try again.")
    
    try:
        start_time = datetime.now()
        
        # Prepare the prompt
        system_prompt = (
            "You are Moses Omondi's personal AI recruitment assistant. "
            "Showcase Moses's comprehensive DevSecOps, MLOps, and MLSecOps expertise. "
            "Respond professionally and highlight his qualifications for senior AI and security leadership roles."
        )
        
        # Add context if provided
        if request.context:
            system_prompt += f" Context: {request.context}"
        
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{request.message} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = full_response.find("[/INST]") + 7
        response = full_response[response_start:].strip()
        
        # Clean up response
        if response.endswith("</s>"):
            response = response[:-4].strip()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return ChatResponse(
            response=response,
            timestamp=end_time.isoformat(),
            model_info={
                "model": "Moses Omondi's AI Recruitment Assistant",
                "base_model": "Llama-2-7b-chat-hf",
                "specialization": "DevSecOps + MLOps + MLSecOps",
                "version": "1.0.0"
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/capabilities")
async def get_capabilities():
    """Get information about Moses's capabilities and expertise areas."""
    return {
        "expertise_areas": [
            "DevSecOps Leadership & Transformation",
            "MLOps at Enterprise Scale", 
            "MLSecOps & AI Security",
            "Cloud Security Architecture",
            "AI/ML Engineering & Leadership",
            "Executive Technical Leadership"
        ],
        "role_types": [
            "VP of AI Security / Chief AI Security Officer",
            "VP of MLOps / Chief ML Officer", 
            "Principal DevSecOps Engineer",
            "Senior AI/ML Engineering Leader",
            "CTO (AI-focused companies)",
            "Head of Secure AI"
        ],
        "technical_skills": [
            "Python, Java, Go, JavaScript",
            "PyTorch, TensorFlow, MLflow",
            "Kubernetes, Docker, Terraform", 
            "AWS, Azure, GCP",
            "CI/CD Security, SAST/DAST",
            "Zero-Trust Architecture"
        ],
        "education": {
            "masters": "Missouri State University - MS Computer Science + Data Science Certificate",
            "bachelors": "University of Mysore - BS Computer Science (2018-2021)",
            "honors": "Kanthula Prize Award - Best International Student",
            "location": "Springfield, Missouri"
        }
    }

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions for testing the AI assistant."""
    return {
        "devsecops_questions": [
            "Tell me about Moses's DevSecOps experience and security transformation leadership.",
            "How can Moses help implement security automation in our CI/CD pipelines?",
            "What's Moses's experience with cloud security and compliance?"
        ],
        "mlops_questions": [
            "What's Moses's MLOps expertise? Can he scale AI operations at enterprise level?",
            "How has Moses implemented ML model deployment and monitoring systems?",
            "Tell me about Moses's experience with production ML infrastructure."
        ],
        "mlsecops_questions": [
            "Does Moses have experience with AI security and MLSecOps?", 
            "How can Moses help secure our machine learning models and data?",
            "What's Moses's approach to AI governance and compliance?"
        ],
        "leadership_questions": [
            "Is Moses qualified for VP-level AI and security roles?",
            "What's Moses's experience leading technical teams?",
            "How has Moses driven digital transformation initiatives?"
        ],
        "general_questions": [
            "What programming languages and technologies does Moses work with?",
            "Tell me about Moses's educational background and certifications.",
            "Is Moses available for remote work or relocation?"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker due to GPU model loading
        log_level="info"
    )
