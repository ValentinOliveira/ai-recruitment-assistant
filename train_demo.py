#!/usr/bin/env python3
"""
Quick Demo Training Script
==========================

Trains a small model quickly for demonstration purposes.
Uses DialoGPT-small instead of Llama for faster training.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("🚀 AI Recruitment Assistant - Quick Demo Training")
print("=" * 60)

# Check if training data exists
data_path = Path("data/training/alpaca_format.json")
if not data_path.exists():
    print("❌ Training data not found!")
    print("Generating training data first...")
    os.system("py src/data/create_sample_data.py --count 50")

print("🤖 Starting quick demo training...")
print("⏱️  This should take about 5-10 minutes...")
print("📊 Using smaller DialoGPT model for speed")

try:
    # Import training components
    from src.config.config_manager import ConfigManager
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import torch

    # Load demo config
    config = ConfigManager(config_path="configs/demo_config.yaml")
    print(f"✅ Configuration loaded")

    # Load tokenizer and model
    print("📥 Loading model and tokenizer...")
    model_name = config.get("model.name")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # Use float32 for demo
    )

    # Load and prepare data
    print("📊 Preparing training data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Take only first 20 examples for demo
    demo_data = data[:20]
    
    # Format for training
    formatted_data = []
    for example in demo_data:
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        formatted_data.append({"text": text})

    # Create dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set up training
    output_dir = Path("models/demo-fine-tuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("🏋️ Starting training...")
    start_time = time.time()
    
    # Train
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Save model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"✅ Demo training completed in {training_time:.1f} seconds!")
    print(f"💾 Model saved to: {output_dir}")
    print("\n🎉 Ready to start the API server!")
    print("Run: start_server.bat")

except Exception as e:
    print(f"❌ Training failed: {e}")
    print("\nTrying simpler approach...")
    
    # Create a dummy model folder for demo
    demo_path = Path("models/demo-fine-tuned")
    demo_path.mkdir(parents=True, exist_ok=True)
    
    # Create a simple marker file
    with open(demo_path / "README.txt", "w") as f:
        f.write("Demo model placeholder - API will use base model")
    
    print("✅ Demo setup complete (using base model)")
    print("🚀 You can now start the server with: start_server.bat")
