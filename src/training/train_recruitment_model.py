#!/usr/bin/env python3
"""
AI Recruitment Assistant - Fine-tuning Script
==============================================

Fine-tunes Llama 3.1 8B model using LoRA for recruitment communication tasks.
Optimized for RTX 4060 (8GB VRAM) with comprehensive monitoring.

Key Features:
- LoRA/QLoRA for memory efficiency
- 4-bit quantization for RTX 4060
- Weights & Biases integration
- TensorBoard logging
- Comprehensive error handling
- Memory optimization
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
import wandb
from accelerate import Accelerator
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 512
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 2  # Optimized for RTX 4060
    gradient_accumulation_steps: int = 4  # Simulate batch size of 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Memory optimization
    use_4bit: bool = True
    use_nested_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Paths
    dataset_path: str = "data/training/alpaca_format.json"
    output_dir: str = "models/checkpoints"
    final_model_dir: str = "models/fine-tuned"
    
    # Monitoring
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    wandb_project: str = "ai-recruitment-assistant"
    wandb_run_name: str = "llama-3.1-8b-lora"

class RecruitmentDataset:
    """Dataset class for recruitment training data."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        
    def load_and_prepare_data(self, tokenizer):
        """Load and prepare the training dataset."""
        self.tokenizer = tokenizer
        
        # Load dataset
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Format data for instruction following
        formatted_data = []
        for item in data:
            # Create instruction-following format
            if item.get("input"):
                text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            formatted_data.append({"text": text})
        
        # Create Hugging Face dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Tokenized dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset

class ModelTrainer:
    """Main training class for the recruitment assistant model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        # Setup directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.final_model_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False  # Disable for training
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure model for training
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info("✅ Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        logger.info("Setting up LoRA configuration...")
        
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("✅ LoRA configuration applied")
    
    def setup_data(self):
        """Load and prepare training dataset."""
        logger.info("Loading training dataset...")
        
        dataset_loader = RecruitmentDataset(self.config)
        self.dataset = dataset_loader.load_and_prepare_data(self.tokenizer)
        
        logger.info(f"✅ Dataset prepared with {len(self.dataset)} examples")
    
    def setup_monitoring(self):
        """Initialize Weights & Biases monitoring."""
        try:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
                tags=["llama-3.1", "lora", "recruitment", "rtx-4060"]
            )
            logger.info("✅ Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            logger.info("Continuing without W&B monitoring")
    
    def train(self):
        """Execute the training process."""
        logger.info("🚀 Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim="adamw_bnb_8bit",  # Memory efficient optimizer
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=False,
            bf16=True,  # Better for training stability
            max_grad_norm=1.0,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if wandb.run else None,
            run_name=self.config.wandb_run_name,
            
            # Memory optimization
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            
            # Logging and saving
            save_total_limit=2,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        try:
            result = trainer.train()
            
            logger.info("🎉 Training completed successfully!")
            logger.info(f"Training loss: {result.training_loss:.4f}")
            
            # Save the final model
            self.save_model()
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the fine-tuned model."""
        logger.info("💾 Saving fine-tuned model...")
        
        try:
            # Save LoRA adapter
            self.model.save_pretrained(self.config.final_model_dir)
            self.tokenizer.save_pretrained(self.config.final_model_dir)
            
            # Save training config
            config_path = Path(self.config.final_model_dir) / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info(f"✅ Model saved to: {self.config.final_model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B for recruitment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Display configuration
    logger.info("🔧 Training Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Please check your GPU setup.")
        sys.exit(1)
    
    gpu_info = torch.cuda.get_device_properties(0)
    logger.info(f"🔥 GPU: {gpu_info.name}")
    logger.info(f"📊 VRAM: {gpu_info.total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    try:
        # Setup monitoring
        trainer.setup_monitoring()
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        # Setup LoRA
        trainer.setup_lora()
        
        # Load data
        trainer.setup_data()
        
        # Train model
        result = trainer.train()
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()
