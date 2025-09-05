#!/usr/bin/env python3
"""
MacBook Client for AI Recruitment Assistant
===========================================

Simple client script to interact with the AI Recruitment Assistant API
from your MacBook. Provides a command-line interface for generating
recruitment communications.

Usage:
    python3 macbook_client.py --server 192.168.1.100:8000
"""

import requests
import json
import argparse
import sys
from typing import Dict, Any

class RecruitmentAssistantClient:
    """Client for accessing the AI Recruitment Assistant API."""
    
    def __init__(self, server_url: str):
        self.base_url = f"http://{server_url}"
        self.session = requests.Session()
        
    def check_health(self) -> bool:
        """Check if the server is healthy and accessible."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy")
                print(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
                print(f"   GPU available: {health_data.get('gpu_available', 'Unknown')}")
                print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f} seconds")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            print(f"   Make sure the server is running at: {self.base_url}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get model info: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting model info: {e}")
            return {}
    
    def generate_response(
        self, 
        instruction: str, 
        input_text: str = "", 
        temperature: float = 0.7,
        max_length: int = 256
    ) -> str:
        """Generate a response using the AI assistant."""
        
        payload = {
            "instruction": instruction,
            "input": input_text,
            "temperature": temperature,
            "max_length": max_length
        }
        
        try:
            print("🤖 Generating response...")
            response = self.session.post(
                f"{self.base_url}/generate", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"⏱️  Generated in {result.get('generation_time', 0):.2f} seconds")
                return result["response"]
            else:
                error_msg = response.json().get("detail", "Unknown error")
                print(f"❌ Generation failed: {error_msg}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return ""
    
    def interactive_mode(self):
        """Run in interactive chat mode."""
        print("🚀 AI Recruitment Assistant - Interactive Mode")
        print("=" * 50)
        print("Enter your recruitment communication requests.")
        print("Type 'quit' to exit, 'help' for examples.")
        print("=" * 50)
        
        while True:
            print("\n" + "─" * 50)
            
            # Get instruction
            instruction = input("📝 What do you need help with? ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if instruction.lower() == 'help':
                self.show_examples()
                continue
                
            if not instruction:
                print("Please enter an instruction.")
                continue
            
            # Get optional input
            input_text = input("📄 Additional context (optional): ").strip()
            
            # Generate response
            response = self.generate_response(instruction, input_text)
            
            if response:
                print("\n✅ Generated Response:")
                print("─" * 40)
                print(response)
                print("─" * 40)
            
    def show_examples(self):
        """Show example prompts."""
        examples = [
            {
                "title": "Interview Scheduling",
                "instruction": "Schedule an interview with a candidate who applied for a Software Engineer position.",
                "input": "Candidate: John Doe\nPosition: Software Engineer\nAvailable times: This week afternoons"
            },
            {
                "title": "Application Status Update", 
                "instruction": "Respond to a candidate asking about their application status.",
                "input": "Candidate applied for Data Scientist role 2 weeks ago"
            },
            {
                "title": "Job Offer",
                "instruction": "Send a job offer to a successful candidate.",
                "input": "Candidate: Jane Smith\nPosition: Product Manager\nSalary: $120k"
            },
            {
                "title": "Job Description",
                "instruction": "Create a job description for a UX Designer position.",
                "input": "Company: TechCorp\nSkills: Figma, User Research, Prototyping"
            }
        ]
        
        print("\n💡 Example Prompts:")
        print("=" * 40)
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['title']}")
            print(f"   Instruction: {example['instruction']}")
            print(f"   Context: {example['input']}")
    
    def batch_test(self):
        """Run a quick batch test with predefined examples."""
        print("🧪 Running batch test...")
        
        test_cases = [
            {
                "instruction": "Schedule an interview with a candidate.",
                "input": "Candidate: Alice Johnson\nPosition: Data Scientist\nTime: Tomorrow 2 PM"
            },
            {
                "instruction": "Send a thank you email after an interview.",
                "input": "Candidate interviewed for Software Engineer position yesterday"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*20} Test {i} {'='*20}")
            print(f"Instruction: {test['instruction']}")
            print(f"Input: {test['input']}")
            
            response = self.generate_response(test["instruction"], test["input"])
            if response:
                print(f"Response: {response[:200]}...")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AI Recruitment Assistant MacBook Client")
    parser.add_argument("--server", "-s", type=str, required=True,
                       help="Server address (e.g., 192.168.1.100:8000)")
    parser.add_argument("--mode", "-m", choices=["interactive", "test", "health"],
                       default="interactive", help="Operation mode")
    
    args = parser.parse_args()
    
    # Initialize client
    client = RecruitmentAssistantClient(args.server)
    
    # Check server health first
    print(f"🔍 Connecting to server at: {args.server}")
    if not client.check_health():
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure the Windows server is running: start_server.bat")
        print("2. Check the IP address is correct")
        print("3. Ensure both devices are on the same network")
        print("4. Check Windows Firewall settings")
        sys.exit(1)
    
    # Show model info
    model_info = client.get_model_info()
    if model_info:
        print(f"\n📊 Model Information:")
        print(f"   Model: {model_info.get('model_name', 'Unknown')}")
        print(f"   LoRA enabled: {model_info.get('lora_enabled', False)}")
        print(f"   Device: {model_info.get('device', 'Unknown')}")
    
    # Run requested mode
    if args.mode == "interactive":
        client.interactive_mode()
    elif args.mode == "test":
        client.batch_test()
    elif args.mode == "health":
        print("✅ Health check completed")

if __name__ == "__main__":
    main()
