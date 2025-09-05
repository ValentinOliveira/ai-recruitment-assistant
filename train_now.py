#!/usr/bin/env python3
"""
Immediate Training Script - No Authentication Required
=====================================================

Start training immediately using GPT-2 XL - no access restrictions.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

print("[STARTING] TRAINING NOW - RTX 4060 OPTIMIZED")
print("=" * 60)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
Path("models/fine-tuned").mkdir(parents=True, exist_ok=True)

try:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    logger.info("[SUCCESS] All packages imported successfully")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.error("[ERROR] No CUDA GPU available!")
        sys.exit(1)
    
    # Load training data
    logger.info("[DATA] Loading training data...")
    with open("data/training/alpaca_format.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"[DATA] Loaded {len(data)} training examples")
    
    # Use GPT-2 XL - no authentication needed
    MODEL_NAME = "gpt2-xl"  # 1.5B parameters, perfect for RTX 4060
    
    logger.info(f"[MODEL] Loading model: {MODEL_NAME}")
    
    # Configure for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("[SUCCESS] Model and tokenizer loaded")
    
    # Configure LoRA for GPT-2
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradients for training with quantized model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    
    logger.info("[SUCCESS] LoRA configuration applied")
    
    # Prepare training data
    def format_example(example):
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}
    
    # Format data
    formatted_data = [format_example(ex) for ex in data]
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    logger.info("[SUCCESS] Data prepared and tokenized")
    
    # Training arguments optimized for RTX 4060
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Conservative for 8GB VRAM
        gradient_accumulation_steps=8,   # Effective batch size = 8
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",  # Updated parameter name
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        bf16=True,                    # Use bfloat16
        remove_unused_columns=False,
        report_to=None,               # Disable W&B for now
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(20)),  # Small eval set
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("[TRAINING] STARTING TRAINING - RTX 4060 OPTIMIZED")
    logger.info("[CONFIG] Configuration:")
    logger.info(f"   Model: {MODEL_NAME}")
    logger.info(f"   Training examples: {len(data)}")
    logger.info(f"   Epochs: 3")
    logger.info(f"   Batch size: 1 (effective: 8 with accumulation)")
    logger.info(f"   Learning rate: 2e-4")
    logger.info(f"   Memory optimization: 4-bit + LoRA + bf16")
    
    start_time = time.time()
    
    # Train the model
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    logger.info(f"[COMPLETED] TRAINING COMPLETED!")
    logger.info(f"[TIME] Total time: {training_time/60:.1f} minutes")
    logger.info(f"[LOSS] Final loss: {train_result.training_loss:.4f}")
    
    # Save the model
    logger.info("[SAVE] Saving fine-tuned model...")
    trainer.save_model("models/fine-tuned")
    tokenizer.save_pretrained("models/fine-tuned")
    
    # Save training config
    with open("models/fine-tuned/training_info.json", "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "training_time_minutes": training_time / 60,
            "final_loss": train_result.training_loss,
            "training_examples": len(data),
            "epochs": 3,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info("[SUCCESS] MODEL SAVED TO: models/fine-tuned")
    logger.info("[COMPLETE] Training complete! You can now:")
    logger.info("   1. Test the model: py src/inference/test_model.py")
    logger.info("   2. Start API server: py src/deployment/api_server.py")
    logger.info("   3. Monitor via SSH from MacBook")

except Exception as e:
    logger.error(f"[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
