#!/usr/bin/env python3
"""
AI Recruitment Assistant - Model Inference and Testing
======================================================

Test and interact with the fine-tuned Llama 3.1 8B recruitment assistant model.
Supports both interactive chat mode and batch testing with evaluation metrics.

Features:
- Interactive chat interface
- Batch testing with predefined scenarios
- Response quality evaluation
- Performance benchmarking
- Model comparison utilities
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecruitmentAssistantTester:
    """Test and interact with the fine-tuned recruitment assistant model."""
    
    def __init__(self, model_path: str, base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Test scenarios
        self.test_scenarios = [
            {
                "category": "Interview Scheduling",
                "instruction": "Schedule an interview with a candidate who applied for a Software Engineer position.",
                "input": "Candidate: Sarah Johnson\nPosition: Software Engineer\nAvailable times: Tuesday/Wednesday afternoons\nInterview type: Technical",
                "expected_elements": ["Dear Sarah", "Software Engineer", "interview", "Tuesday", "Wednesday"]
            },
            {
                "category": "Application Status",
                "instruction": "Respond to a candidate who inquired about the status of their application.",
                "input": "Candidate Michael Chen applied for Data Scientist position 3 weeks ago and is asking for an update",
                "expected_elements": ["Dear Michael", "Data Scientist", "application", "review", "update"]
            },
            {
                "category": "Job Offer",
                "instruction": "Send a job offer to a successful candidate.",
                "input": "Candidate Emily Rodriguez has been selected for the Frontend Developer position with salary $95k",
                "expected_elements": ["Congratulations", "Emily", "Frontend Developer", "offer", "$95"]
            },
            {
                "category": "Rejection Letter",
                "instruction": "Send a rejection email to a candidate after the interview process.",
                "input": "Candidate David Park interviewed for Product Manager but was not selected",
                "expected_elements": ["Dear David", "Product Manager", "thank you", "decision", "future"]
            },
            {
                "category": "Job Description",
                "instruction": "Create a job description for a DevOps Engineer position.",
                "input": "Company: TechCorp\nPosition: DevOps Engineer\nKey skills: AWS, Docker, Kubernetes",
                "expected_elements": ["DevOps Engineer", "AWS", "Docker", "Kubernetes", "responsibilities"]
            }
        ]
    
    def load_model(self, use_4bit: bool = True):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Configure quantization for inference
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"  # Left padding for generation
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info("✅ LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter: {e}")
            logger.info("Using base model instead")
            self.model = base_model
        
        # Create generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        logger.info("✅ Model loaded and ready for inference")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt in the training format."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(
        self, 
        instruction: str, 
        input_text: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate a response from the model."""
        prompt = self.format_prompt(instruction, input_text)
        
        # Generate response
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extract generated text
        generated_text = outputs[0]['generated_text']
        
        # Clean up the response
        response = generated_text.split("### Response:")[-1].strip()
        
        # Remove any remaining special tokens
        response = response.replace("<|end_of_text|>", "").strip()
        
        return response
    
    def evaluate_response(self, response: str, expected_elements: List[str]) -> Dict[str, Any]:
        """Evaluate response quality based on expected elements."""
        response_lower = response.lower()
        
        # Check for expected elements
        found_elements = []
        missing_elements = []
        
        for element in expected_elements:
            if element.lower() in response_lower:
                found_elements.append(element)
            else:
                missing_elements.append(element)
        
        # Calculate scores
        element_score = len(found_elements) / len(expected_elements) if expected_elements else 1.0
        length_score = min(len(response) / 200, 1.0)  # Prefer responses around 200 chars
        
        # Basic quality checks
        has_greeting = any(greeting in response_lower for greeting in ["dear", "hi", "hello"])
        has_closing = any(closing in response_lower for closing in ["regards", "sincerely", "thank you", "best"])
        is_professional = not any(word in response_lower for word in ["lol", "haha", "omg"])
        
        quality_score = sum([has_greeting, has_closing, is_professional]) / 3.0
        
        overall_score = (element_score * 0.5 + length_score * 0.2 + quality_score * 0.3)
        
        return {
            "overall_score": overall_score,
            "element_score": element_score,
            "length_score": length_score,
            "quality_score": quality_score,
            "found_elements": found_elements,
            "missing_elements": missing_elements,
            "has_greeting": has_greeting,
            "has_closing": has_closing,
            "is_professional": is_professional,
            "response_length": len(response)
        }
    
    def run_batch_tests(self) -> pd.DataFrame:
        """Run batch tests on predefined scenarios."""
        logger.info("🧪 Running batch tests...")
        
        results = []
        
        for i, scenario in enumerate(self.test_scenarios):
            logger.info(f"Testing scenario {i+1}/{len(self.test_scenarios)}: {scenario['category']}")
            
            start_time = time.time()
            
            # Generate response
            response = self.generate_response(
                scenario["instruction"],
                scenario["input"]
            )
            
            generation_time = time.time() - start_time
            
            # Evaluate response
            evaluation = self.evaluate_response(response, scenario["expected_elements"])
            
            # Compile results
            result = {
                "scenario_id": i + 1,
                "category": scenario["category"],
                "instruction": scenario["instruction"],
                "input": scenario["input"],
                "response": response,
                "generation_time": generation_time,
                **evaluation
            }
            
            results.append(result)
            
            # Show progress
            logger.info(f"  Score: {evaluation['overall_score']:.2f} | Time: {generation_time:.2f}s")
        
        df_results = pd.DataFrame(results)
        
        # Summary statistics
        logger.info("\n📊 Batch Test Results:")
        logger.info(f"Average Overall Score: {df_results['overall_score'].mean():.3f}")
        logger.info(f"Average Generation Time: {df_results['generation_time'].mean():.2f}s")
        logger.info(f"Best Category: {df_results.loc[df_results['overall_score'].idxmax(), 'category']}")
        logger.info(f"Worst Category: {df_results.loc[df_results['overall_score'].idxmin(), 'category']}")
        
        return df_results
    
    def interactive_chat(self):
        """Run interactive chat mode."""
        logger.info("🚀 Interactive Chat Mode - Type 'quit' to exit")
        logger.info("You can provide both instruction and input, or just instruction.")
        
        while True:
            print("\n" + "="*60)
            instruction = input("📝 Instruction: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                break
            
            input_text = input("📄 Input (optional, press Enter to skip): ").strip()
            
            print("\n🤖 Generating response...")
            start_time = time.time()
            
            try:
                response = self.generate_response(instruction, input_text)
                generation_time = time.time() - start_time
                
                print(f"\n✅ Response (generated in {generation_time:.2f}s):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Quick evaluation
                evaluation = self.evaluate_response(response, [])
                print(f"\n📊 Quick Stats:")
                print(f"  • Length: {evaluation['response_length']} characters")
                print(f"  • Has greeting: {'✅' if evaluation['has_greeting'] else '❌'}")
                print(f"  • Has closing: {'✅' if evaluation['has_closing'] else '❌'}")
                print(f"  • Professional tone: {'✅' if evaluation['is_professional'] else '❌'}")
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
    
    def benchmark_performance(self, num_tests: int = 10) -> Dict[str, float]:
        """Benchmark model performance."""
        logger.info(f"🏃 Running performance benchmark with {num_tests} tests...")
        
        test_prompt = "Schedule an interview with a candidate who applied for a Software Engineer position."
        test_input = "Candidate: John Doe\nPosition: Software Engineer\nAvailable times: This week"
        
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            _ = self.generate_response(test_prompt, test_input)
            times.append(time.time() - start_time)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  Completed {i + 1}/{num_tests} tests...")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"\n⚡ Performance Results:")
        logger.info(f"  • Average: {avg_time:.2f}s")
        logger.info(f"  • Fastest: {min_time:.2f}s")
        logger.info(f"  • Slowest: {max_time:.2f}s")
        
        return {
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_tests": num_tests
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the AI Recruitment Assistant model")
    parser.add_argument("--model-path", type=str, default="models/fine-tuned",
                       help="Path to the fine-tuned model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base model name")
    parser.add_argument("--mode", type=str, choices=["chat", "test", "benchmark", "all"],
                       default="chat", help="Mode to run")
    parser.add_argument("--output", type=str, help="Output file for test results")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        logger.info("Please train the model first using:")
        logger.info("  python src/training/train_recruitment_model.py")
        sys.exit(1)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("⚠️  No GPU detected - inference will be slow")
    
    # Initialize tester
    tester = RecruitmentAssistantTester(args.model_path, args.base_model)
    
    try:
        # Load model
        tester.load_model(use_4bit=not args.no_4bit)
        
        # Run based on mode
        if args.mode == "chat":
            tester.interactive_chat()
        
        elif args.mode == "test":
            results_df = tester.run_batch_tests()
            if args.output:
                results_df.to_csv(args.output, index=False)
                logger.info(f"💾 Results saved to: {args.output}")
        
        elif args.mode == "benchmark":
            benchmark_results = tester.benchmark_performance()
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
                logger.info(f"💾 Benchmark saved to: {args.output}")
        
        elif args.mode == "all":
            logger.info("Running all tests...")
            
            # Batch tests
            results_df = tester.run_batch_tests()
            
            # Benchmark
            benchmark_results = tester.benchmark_performance()
            
            # Save results
            if args.output:
                base_path = Path(args.output)
                
                # Save test results
                test_path = base_path.with_suffix('.csv')
                results_df.to_csv(test_path, index=False)
                
                # Save benchmark
                bench_path = base_path.with_stem(f"{base_path.stem}_benchmark").with_suffix('.json')
                with open(bench_path, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
                
                logger.info(f"💾 Results saved to: {test_path} and {bench_path}")
            
            # Interactive mode
            logger.info("\nSwitching to interactive mode...")
            tester.interactive_chat()
    
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
