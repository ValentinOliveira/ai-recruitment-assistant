#!/usr/bin/env python3
"""
Moses Omondi's Personal AI Recruitment Assistant - Final Training
================================================================

Train Moses's personal AI assistant using Llama 2 7B with the comprehensive
MLOps/MLSecOps dataset. This AI will represent Moses to recruiters.
"""

import json
import os
import sys
import time
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_moses_ai():
    """Train Moses's AI recruitment assistant."""
    
    print("🚀 Moses Omondi's Personal AI Recruitment Assistant Training")
    print("=" * 60)
    print("This AI will represent Moses to recruiters and companies")
    print("Dataset: MLOps + MLSecOps + AI Leadership + Authentic Profile")
    print("")
    
    # Configuration
    MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
    OUTPUT_DIR = "models/moses-recruitment-assistant"
    DATASET_PATH = "data/moses_mlops_enhanced_dataset.json"
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ No GPU available!")
        return False
    
    # Load dataset
    print("📊 Loading Moses's comprehensive dataset...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} training examples")
        print("   📈 MLOps/MLSecOps Leadership examples")
        print("   🤖 AI/ML expertise positioning") 
        print("   📱 LinkedIn/GitHub authentic data")
        print("   💼 Professional recruitment scenarios")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Prepare training texts
    print("\n🔄 Preparing training data...")
    texts = []
    for item in data:
        # Format as Llama 2 chat template
        text = f"<s>[INST] <<SYS>>\n{item['instruction']}\n<</SYS>>\n\n{item['input']} [/INST] {item['output']} </s>"
        texts.append(text)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    print(f"✅ Dataset prepared with {len(texts)} examples")
    
    # Model configuration for RTX 4060
    print("\n⚙️ Configuring model for RTX 4060...")
    
    # 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Smaller for memory efficiency
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Model configured and ready")
    
    # Tokenize dataset
    print("\n🔤 Tokenizing dataset...")
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,  # Smaller for memory efficiency
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    print(f"✅ Tokenization complete")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # Conservative for initial training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size of 8
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=2,
        save_steps=10,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=True,
        report_to=None,  # Disable wandb
        seed=42,
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
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"\n🎯 Starting training...")
    print(f"   📊 Examples: {len(tokenized_dataset)}")
    print(f"   🔄 Epochs: 2")
    print(f"   💾 Batch size: 1 (effective: 8)")
    print(f"   🧠 Model: {MODEL_NAME}")
    print(f"   💼 Focus: Moses's MLOps/MLSecOps expertise")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save training info
        metadata = {
            "model_name": MODEL_NAME,
            "training_time_minutes": training_time / 60,
            "dataset_size": len(data),
            "epochs": 2,
            "description": "Moses Omondi's Personal AI Recruitment Assistant",
            "capabilities": [
                "MLOps and MLSecOps expertise",
                "AI/ML leadership positioning", 
                "Authentic GitHub/LinkedIn profile",
                "Executive-level recruitment conversations",
                "Springfield, Missouri location",
                "VP-level AI role qualifications"
            ]
        }
        
        with open(f"{OUTPUT_DIR}/moses_assistant_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to: {OUTPUT_DIR}")
        print("\n🎉 SUCCESS!")
        print("🤖 Moses's AI Recruitment Assistant is ready!")
        print("\n💼 This AI can now represent Moses for:")
        print("   • VP of MLOps / Chief ML Officer roles")
        print("   • MLSecOps leadership positions") 
        print("   • Senior AI/ML engineering roles")
        print("   • Enterprise AI transformation opportunities")
        print("   • Technical discussions with FAANG recruiters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_moses_ai()
    sys.exit(0 if success else 1)
