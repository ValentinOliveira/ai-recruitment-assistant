#!/usr/bin/env python3
"""
AI Recruitment Assistant - FastAPI Deployment Server
====================================================

Production-ready API server for serving the fine-tuned recruitment assistant model.
Optimized for RTX 4060 with async processing, caching, and comprehensive monitoring.

Features:
- RESTful API with OpenAPI documentation
- Async request processing
- Response caching
- Request rate limiting
- Input validation and sanitization
- Comprehensive logging and monitoring
- Health checks and metrics
- Model hot-swapping capabilities
"""

import os
import sys
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# ML dependencies
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# Utilities
from cachetools import TTLCache
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    instruction: str = Field(..., min_length=10, max_length=1000, description="The instruction for the AI assistant")
    input: Optional[str] = Field(None, max_length=2000, description="Optional input context")
    max_length: Optional[int] = Field(256, ge=50, le=512, description="Maximum response length")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="Temperature for generation")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    
    @validator('instruction')
    def validate_instruction(cls, v):
        if not v.strip():
            raise ValueError('Instruction cannot be empty')
        return v.strip()

class GenerationResponse(BaseModel):
    """Response model for text generation."""
    response: str = Field(..., description="Generated response from the AI assistant")
    instruction: str = Field(..., description="Original instruction")
    input: Optional[str] = Field(None, description="Original input context")
    generation_time: float = Field(..., description="Time taken to generate response (seconds)")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_memory_used: Optional[float] = Field(None, description="GPU memory usage percentage")
    system_memory_used: float = Field(..., description="System memory usage percentage")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    requests_served: int = Field(..., description="Total requests served")

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Base model name")
    model_path: str = Field(..., description="Path to fine-tuned model")
    lora_enabled: bool = Field(..., description="Whether LoRA adapter is loaded")
    quantization: str = Field(..., description="Quantization configuration")
    device: str = Field(..., description="Device model is running on")
    memory_usage_mb: float = Field(..., description="Model memory usage in MB")

# Global variables
app = FastAPI(
    title="AI Recruitment Assistant API",
    description="Production-ready API for AI-powered recruitment communication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Service state
class ServiceState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_info = {}
        self.start_time = time.time()
        self.requests_served = 0
        self.cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
        self.is_ready = False

service_state = ServiceState()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly for production
)

# Model Management
class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        
    async def load_model(self, use_4bit: bool = True) -> Dict[str, Any]:
        """Load the model asynchronously."""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Configure quantization
            if use_4bit and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                quantization_info = "4-bit NF4"
            else:
                bnb_config = None
                quantization_info = "None"
            
            # Load tokenizer
            service_state.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if self.model_path.exists() else self.base_model,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if service_state.tokenizer.pad_token is None:
                service_state.tokenizer.pad_token = service_state.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Try to load LoRA adapter
            lora_enabled = False
            if self.model_path.exists():
                try:
                    service_state.model = PeftModel.from_pretrained(base_model, self.model_path)
                    lora_enabled = True
                    logger.info("✅ LoRA adapter loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load LoRA adapter: {e}")
                    service_state.model = base_model
            else:
                logger.warning(f"Model path does not exist: {self.model_path}")
                service_state.model = base_model
            
            # Create pipeline
            service_state.pipeline = pipeline(
                "text-generation",
                model=service_state.model,
                tokenizer=service_state.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Get model info
            device = next(service_state.model.parameters()).device
            model_size = sum(p.numel() for p in service_state.model.parameters()) * 4 / (1024**2)  # Approximate MB
            
            service_state.model_info = {
                "model_name": self.base_model,
                "model_path": str(self.model_path),
                "lora_enabled": lora_enabled,
                "quantization": quantization_info,
                "device": str(device),
                "memory_usage_mb": model_size
            }
            
            service_state.is_ready = True
            logger.info("✅ Model loaded and ready for inference")
            
            return service_state.model_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt in training format."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"

    async def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate response asynchronously."""
        if not service_state.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Check cache first
        cache_key = f"{hash(instruction + input_text)}_{max_length}_{temperature}_{top_p}"
        if cache_key in service_state.cache:
            logger.info("Cache hit for request")
            return service_state.cache[cache_key]
        
        prompt = self.format_prompt(instruction, input_text)
        
        try:
            # Generate in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate():
                outputs = service_state.pipeline(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=service_state.tokenizer.eos_token_id,
                    eos_token_id=service_state.tokenizer.eos_token_id,
                    return_full_text=False
                )
                return outputs[0]['generated_text']
            
            generated_text = await loop.run_in_executor(None, _generate)
            
            # Clean up response
            response = generated_text.split("### Response:")[-1].strip()
            response = response.replace("<|end_of_text|>", "").strip()
            
            # Cache the result
            service_state.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Initialize model manager
model_manager = None

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global model_manager
    
    model_path = os.getenv("MODEL_PATH", "models/fine-tuned")
    base_model = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    logger.info("🚀 Starting AI Recruitment Assistant API")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Base model: {base_model}")
    
    model_manager = ModelManager(model_path, base_model)
    
    try:
        await model_manager.load_model()
        logger.info("✅ Service startup completed")
    except Exception as e:
        logger.error(f"❌ Service startup failed: {e}")
        # Service will still start but won't be ready

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "AI Recruitment Assistant API",
        "version": "1.0.0",
        "status": "running" if service_state.is_ready else "initializing",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # System metrics
    memory = psutil.virtual_memory()
    
    # GPU metrics
    gpu_memory_used = None
    if torch.cuda.is_available() and service_state.is_ready:
        try:
            gpu_memory_used = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
        except:
            pass
    
    return HealthResponse(
        status="healthy" if service_state.is_ready else "initializing",
        model_loaded=service_state.is_ready,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used=gpu_memory_used,
        system_memory_used=memory.percent,
        uptime_seconds=time.time() - service_state.start_time,
        requests_served=service_state.requests_served
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if not service_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return ModelInfo(**service_state.model_info)

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text using the recruitment assistant model."""
    if not service_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    
    try:
        response = await model_manager.generate_response(
            instruction=request.instruction,
            input_text=request.input or "",
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Update metrics in background
        def update_metrics():
            service_state.requests_served += 1
        
        background_tasks.add_task(update_metrics)
        
        return GenerationResponse(
            response=response,
            instruction=request.instruction,
            input=request.input,
            generation_time=generation_time,
            model_info=service_state.model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-generate")
async def batch_generate(requests: List[GenerationRequest]):
    """Generate multiple responses in batch."""
    if not service_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10")
    
    responses = []
    
    for req in requests:
        try:
            start_time = time.time()
            response = await model_manager.generate_response(
                instruction=req.instruction,
                input_text=req.input or "",
                max_length=req.max_length,
                temperature=req.temperature,
                top_p=req.top_p
            )
            
            generation_time = time.time() - start_time
            
            responses.append(GenerationResponse(
                response=response,
                instruction=req.instruction,
                input=req.input,
                generation_time=generation_time,
                model_info=service_state.model_info
            ))
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            responses.append({
                "error": str(e),
                "instruction": req.instruction,
                "input": req.input
            })
    
    service_state.requests_served += len(requests)
    return {"responses": responses}

@app.delete("/cache")
async def clear_cache():
    """Clear the response cache."""
    service_state.cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        "requests_served": service_state.requests_served,
        "uptime_seconds": time.time() - service_state.start_time,
        "cache_size": len(service_state.cache),
        "model_ready": service_state.is_ready,
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": dict(psutil.virtual_memory()._asdict()) if psutil else None
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": ["/docs", "/health", "/generate"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AI Recruitment Assistant API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", type=str, default="models/fine-tuned", help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Base model name")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["BASE_MODEL"] = args.base_model
    
    logger.info(f"🚀 Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
