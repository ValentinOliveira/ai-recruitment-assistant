#!/usr/bin/env python3
"""
Simple test of Moses's updated education information
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def test_education():
    """Test the updated education information."""
    
    print("🎓 Testing Moses's Updated Educational Background")
    print("=" * 60)
    
    MODEL_PATH = "models/moses-recruitment-assistant"
    BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
    
    # 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print("📚 Loading model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    print("✅ Model loaded successfully!")
    print()
    
    # Test education question
    question = "Tell me about Moses Omondi's educational background and degrees."
    
    print("❓ Testing Question:")
    print(f"   {question}")
    print()
    
    system_prompt = "You are Moses Omondi's personal AI recruitment assistant. Respond professionally about Moses's educational qualifications."
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question} [/INST]"
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🤖 Generating response...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode and display response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = full_response.find("[/INST]") + 7
    response = full_response[response_start:].strip()
    
    print("💬 MOSES'S AI ASSISTANT RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    print()
    
    # Check if key education details are present
    print("🔍 EDUCATION VERIFICATION:")
    checks = {
        "Missouri State University": "Missouri State University" in response,
        "University of Mysore": "University of Mysore" in response, 
        "Master of Science": "Master of Science" in response or "MS" in response,
        "Computer Science": "Computer Science" in response,
        "Data Science": "Data Science" in response,
        "Kanthula Prize": "Kanthula Prize" in response,
        "Best International Student": "Best International Student" in response or "International Student" in response,
        "2018-2021": "2018" in response or "2021" in response
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {'Found' if passed else 'Not Found'}")
    
    passed_checks = sum(checks.values())
    print(f"\n📊 Overall: {passed_checks}/{len(checks)} checks passed")
    
    if passed_checks >= len(checks) * 0.8:
        print("🎉 SUCCESS! Education information updated successfully!")
    else:
        print("⚠️  Some education details may need attention.")

if __name__ == "__main__":
    test_education()
