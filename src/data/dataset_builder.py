#!/usr/bin/env python3
"""
AI Recruitment Assistant Dataset Builder
==========================================

This module creates a fine-tuning dataset for the recruitment assistant
using personal profile information and synthetic data generation.

Strategy:
1. Use your LinkedIn/GitHub profile as the foundation
2. Generate synthetic recruitment scenarios
3. Create instruction-following dataset format
4. Focus on professional communication patterns
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile information for personalized responses."""
    name: str
    degree: str
    university: str
    skills: List[str]
    experience_years: int
    current_role: str
    linkedin_url: str
    github_url: str
    location: str
    industry_focus: List[str]

class DatasetBuilder:
    """Builds training dataset for recruitment assistant fine-tuning."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.dataset = []
        
    def create_user_profile_template(self) -> UserProfile:
        """Template for user profile - customize with your information."""
        return UserProfile(
            name="Wesley B",  # Replace with your name
            degree="Master's Degree",
            university="Missouri State University",
            skills=[
                "Software Engineering", "Python", "JavaScript", "React",
                "Node.js", "AWS", "Docker", "Machine Learning", "AI",
                "System Design", "Database Design", "API Development",
                "DevOps", "MLOps", "Data Science", "Deep Learning"
            ],
            experience_years=5,  # Adjust based on your experience
            current_role="Software Engineer",  # Update with your current role
            linkedin_url="linkedin.com/in/your-profile",  # Your LinkedIn
            github_url="github.com/your-username",  # Your GitHub
            location="United States",  # Your location
            industry_focus=[
                "Technology", "Software Development", "Machine Learning",
                "Artificial Intelligence", "Data Science", "Cloud Computing"
            ]
        )
    
    def generate_recruiter_outreach_scenarios(self) -> List[Dict[str, str]]:
        """Generate synthetic recruiter outreach scenarios."""
        
        scenarios = [
            # Initial recruiter contact
            {
                "instruction": "Respond to a recruiter reaching out about a software engineering position",
                "input": f"Hi {self.user_profile.name}, I came across your profile and I'm impressed with your background from {self.user_profile.university}. We have a Senior Software Engineer position that might be a great fit. Are you open to hearing more about it?",
                "output": f"Hi! Thank you for reaching out. I appreciate you taking the time to review my background. I'm always interested in learning about new opportunities, especially those that align with my experience in {', '.join(self.user_profile.skills[:3])}. I'd be happy to hear more about the position and discuss how my skills from my {self.user_profile.degree} and industry experience might be a good fit. Could you share more details about the role and company?"
            },
            
            # Salary discussion
            {
                "instruction": "Discuss salary expectations professionally",
                "input": "What are your salary expectations for this role?",
                "output": f"I'm looking for a competitive compensation package that reflects my {self.user_profile.experience_years} years of experience and {self.user_profile.degree} from {self.user_profile.university}. Based on my research of market rates for similar positions in {self.user_profile.location}, I'd expect a range that's competitive for someone with my background in {', '.join(self.user_profile.skills[:2])}. However, I'm open to discussing the full compensation package including benefits, growth opportunities, and company culture. What range did you have in mind for this position?"
            },
            
            # Scheduling interviews
            {
                "instruction": "Schedule an interview professionally",
                "input": "Great! We'd like to set up an initial phone screening. Are you available this week for a 30-minute call?",
                "output": "That sounds excellent! I'm definitely interested in moving forward with the phone screening. I have availability this week on Tuesday after 2 PM, Wednesday morning between 9 AM and 11 AM, or Thursday afternoon after 1 PM (all times EST). Please let me know which option works best for your schedule, or if you need alternative times. I look forward to learning more about the opportunity and discussing how my background aligns with your needs."
            },
            
            # Following up after applications
            {
                "instruction": "Follow up on a job application professionally",
                "input": "Follow up on the Software Engineer position I applied for two weeks ago",
                "output": f"Subject: Following up on Software Engineer Application\n\nDear Hiring Manager,\n\nI hope this email finds you well. I'm writing to follow up on my application for the Software Engineer position that I submitted two weeks ago. I remain very interested in this opportunity and believe my background in {', '.join(self.user_profile.skills[:3])} and {self.user_profile.degree} from {self.user_profile.university} would be valuable to your team.\n\nI'd welcome the opportunity to discuss how my experience aligns with your needs. Please let me know if you need any additional information or if there's anything I can clarify about my background.\n\nThank you for your time and consideration.\n\nBest regards,\n{self.user_profile.name}"
            },
            
            # Declining opportunities politely
            {
                "instruction": "Politely decline a job opportunity",
                "input": "We'd like to extend an offer for the position, but it's a junior role with a lower salary than expected",
                "output": f"Thank you so much for extending the offer - I'm truly honored that you considered me for this position. After careful consideration, I've decided that this particular role isn't the right fit for my current career goals and experience level. Given my {self.user_profile.experience_years} years of experience and {self.user_profile.degree}, I'm looking for opportunities that better align with my background and career trajectory. I appreciate the time you and your team invested in the interview process, and I hope we might have the chance to work together on a different opportunity in the future. Thank you again for your consideration."
            },
            
            # Asking about company culture
            {
                "instruction": "Ask about company culture and work environment",
                "input": "Do you have any questions about our company?",
                "output": f"Absolutely! I'm very interested in learning more about the company culture and work environment. Could you tell me about the team dynamics and collaboration style? I'm particularly interested in how the engineering team approaches problem-solving and innovation. Also, given my background in {', '.join(self.user_profile.industry_focus[:2])}, I'd love to know about opportunities for professional development and staying current with emerging technologies. What does a typical day look like for someone in this role, and how does the company support work-life balance?"
            },
            
            # Technical discussion
            {
                "instruction": "Discuss technical experience and skills",
                "input": "Can you tell me about your experience with cloud technologies and system architecture?",
                "output": f"Certainly! Throughout my career, I've worked extensively with cloud technologies, particularly AWS services including EC2, S3, Lambda, and RDS. I've designed and implemented scalable system architectures for various applications, focusing on microservices patterns and containerization with Docker. My {self.user_profile.degree} from {self.user_profile.university} provided a strong foundation in system design principles, which I've applied in building distributed systems that handle high traffic loads. I'm particularly passionate about optimizing performance and ensuring reliability in cloud-native applications. Recently, I've been exploring MLOps and AI model deployment on cloud platforms. Would you like me to elaborate on any specific aspect of my cloud or architecture experience?"
            },
            
            # Remote work preferences
            {
                "instruction": "Discuss remote work preferences",
                "input": "This position offers flexible remote work options. What are your preferences?",
                "output": f"That's fantastic to hear! I really appreciate companies that offer flexible work arrangements. I've found that I'm highly productive in both remote and collaborative in-office environments. With my experience in {', '.join(self.user_profile.skills[:2])}, I've developed strong communication and project management skills that translate well to remote work. I'm comfortable with various collaboration tools and asynchronous communication. Ideally, I'd love a hybrid approach that allows for focused deep work remotely while also providing opportunities for in-person collaboration and team building. What does your team's current remote work structure look like?"
            }
        ]
        
        return scenarios
    
    def generate_networking_scenarios(self) -> List[Dict[str, str]]:
        """Generate networking and professional relationship building scenarios."""
        
        scenarios = [
            {
                "instruction": "Respond to a LinkedIn connection request",
                "input": "Someone from your target company sent a LinkedIn connection request",
                "output": f"Hi [Name], thank you for connecting! I noticed you work at [Company] - I've been following the company's work in {', '.join(self.user_profile.industry_focus[:2])} and I'm really impressed with your recent projects. I'd love to learn more about your experience there and the team culture. Perhaps we could grab a virtual coffee sometime to chat about the industry and share experiences?"
            },
            
            {
                "instruction": "Network with industry professionals",
                "input": "Reach out to a software engineering manager for informational interview",
                "output": f"Subject: Informational Interview Request - {self.user_profile.degree} Graduate\n\nHi [Name],\n\nI hope this message finds you well. My name is {self.user_profile.name}, and I'm a software engineer with {self.user_profile.experience_years} years of experience and a {self.user_profile.degree} from {self.user_profile.university}. I've been following [Company]'s innovative work in {self.user_profile.industry_focus[0]}, and I'm very interested in learning more about your team and the engineering culture there.\n\nWould you be open to a brief 15-20 minute informational interview? I'd love to hear about your experience as a manager and learn about potential opportunities for someone with my background in {', '.join(self.user_profile.skills[:3])}.\n\nI understand you're busy, so I'm happy to work around your schedule. Thank you for considering my request!\n\nBest regards,\n{self.user_profile.name}"
            }
        ]
        
        return scenarios
    
    def generate_interview_scenarios(self) -> List[Dict[str, str]]:
        """Generate interview preparation and response scenarios."""
        
        scenarios = [
            {
                "instruction": "Answer 'Tell me about yourself' in an interview",
                "input": "Can you tell me about yourself?",
                "output": f"Certainly! I'm {self.user_profile.name}, a passionate software engineer with {self.user_profile.experience_years} years of experience in {', '.join(self.user_profile.skills[:3])}. I hold a {self.user_profile.degree} from {self.user_profile.university}, which gave me a strong foundation in computer science principles and problem-solving methodologies. Throughout my career, I've focused on building scalable applications and have developed expertise in both frontend and backend technologies. I'm particularly excited about {', '.join(self.user_profile.industry_focus[:2])}, and I love tackling complex technical challenges while collaborating with cross-functional teams. I'm always eager to learn new technologies and contribute to innovative projects that make a real impact."
            },
            
            {
                "instruction": "Explain why you're looking for a new opportunity",
                "input": "Why are you looking for a new role?",
                "output": f"I'm looking for a new opportunity because I'm excited about continuing to grow my career in {self.user_profile.industry_focus[0]}. While I've gained valuable experience in my current role working with {', '.join(self.user_profile.skills[:2])}, I'm seeking an environment where I can take on more challenging projects and have a greater impact on product development. I'm particularly interested in roles that offer opportunities to work with cutting-edge technologies and contribute to innovative solutions. From what I've learned about this position, it seems like an ideal next step that aligns with both my technical skills and career aspirations."
            },
            
            {
                "instruction": "Discuss technical challenges you've overcome",
                "input": "Can you describe a challenging technical problem you've solved?",
                "output": f"One particularly challenging project involved optimizing the performance of a distributed system that was experiencing significant latency issues under high load. The system was built with {self.user_profile.skills[1]} and deployed on AWS, but response times were unacceptable during peak usage. I approached this systematically by first implementing comprehensive monitoring and profiling to identify bottlenecks. I discovered that the issue was a combination of inefficient database queries and inadequate caching strategies. I redesigned the data access layer, implemented Redis caching, and optimized several key algorithms. The result was a 70% reduction in response time and improved system reliability. This experience reinforced my passion for system optimization and the importance of data-driven problem solving."
            }
        ]
        
        return scenarios
    
    def build_dataset(self) -> List[Dict[str, str]]:
        """Build the complete training dataset."""
        
        logger.info("Building recruitment assistant dataset...")
        
        # Collect all scenarios
        scenarios = []
        scenarios.extend(self.generate_recruiter_outreach_scenarios())
        scenarios.extend(self.generate_networking_scenarios())
        scenarios.extend(self.generate_interview_scenarios())
        
        # Add some general professional communication scenarios
        general_scenarios = [
            {
                "instruction": "Provide professional introduction for email",
                "input": "Write a brief professional introduction",
                "output": f"Hello, I'm {self.user_profile.name}, a software engineer with a {self.user_profile.degree} from {self.user_profile.university}. I specialize in {', '.join(self.user_profile.skills[:3])} and have {self.user_profile.experience_years} years of experience building scalable applications and solving complex technical challenges. I'm passionate about {', '.join(self.user_profile.industry_focus[:2])} and always excited to discuss new opportunities and collaborate on innovative projects."
            },
            
            {
                "instruction": "Request a referral professionally",
                "input": "Ask a connection for a referral to their company",
                "output": f"Hi [Name], I hope you're doing well! I've been following [Company]'s work in {self.user_profile.industry_focus[0]} and I'm really impressed with the innovative projects you're working on. I noticed there's a software engineering position open that aligns perfectly with my background in {', '.join(self.user_profile.skills[:3])} and my {self.user_profile.degree} from {self.user_profile.university}. Would you be comfortable providing a referral or introducing me to the hiring manager? I'd be happy to share my resume and discuss how my experience might be a good fit for the team. Thanks for considering it!"
            }
        ]
        
        scenarios.extend(general_scenarios)
        
        # Format for instruction following
        dataset = []
        for scenario in scenarios:
            # Add the scenario as-is
            dataset.append({
                "instruction": scenario["instruction"],
                "input": scenario["input"],
                "output": scenario["output"]
            })
            
            # Add variations to increase dataset size
            if "recruiter" in scenario["instruction"].lower():
                # Create variations for different companies/roles
                variations = self._create_variations(scenario)
                dataset.extend(variations)
        
        self.dataset = dataset
        logger.info(f"Generated {len(dataset)} training examples")
        return dataset
    
    def _create_variations(self, base_scenario: Dict[str, str]) -> List[Dict[str, str]]:
        """Create variations of base scenarios for data augmentation."""
        
        variations = []
        
        # Role variations
        roles = [
            "Senior Software Engineer", "Full Stack Developer", "Backend Engineer",
            "Frontend Engineer", "DevOps Engineer", "Machine Learning Engineer",
            "Data Scientist", "Software Architect", "Technical Lead"
        ]
        
        # Company types
        company_types = [
            "startup", "tech company", "enterprise software company", 
            "fintech company", "healthcare technology company"
        ]
        
        for i, role in enumerate(roles[:3]):  # Limit variations to keep dataset manageable
            variation = base_scenario.copy()
            
            # Modify the input to use different role
            variation["input"] = variation["input"].replace("software engineering position", f"{role} position")
            variation["input"] = variation["input"].replace("Software Engineer", role)
            
            # Modify output accordingly
            variation["output"] = variation["output"].replace("software engineering", role.lower())
            
            variations.append(variation)
            
            if len(variations) >= 2:  # Limit variations per scenario
                break
        
        return variations
    
    def save_dataset(self, output_path: str = "data/training/recruitment_assistant_dataset.json"):
        """Save the dataset to JSON format for training."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_file}")
        logger.info(f"Total examples: {len(self.dataset)}")
        
        # Also save in formats for different training frameworks
        self._save_alpaca_format(output_file.parent / "alpaca_format.json")
        self._save_conversational_format(output_file.parent / "conversational_format.json")
    
    def _save_alpaca_format(self, output_path: Path):
        """Save in Alpaca instruction format."""
        
        alpaca_data = []
        for example in self.dataset:
            alpaca_data.append({
                "instruction": example["instruction"],
                "input": example["input"],
                "output": example["output"]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Alpaca format saved to {output_path}")
    
    def _save_conversational_format(self, output_path: Path):
        """Save in conversational format for chat models."""
        
        conversational_data = []
        for example in self.dataset:
            conversation = {
                "messages": [
                    {"role": "system", "content": f"You are a professional recruitment assistant for {self.user_profile.name}, who has a {self.user_profile.degree} from {self.user_profile.university} and specializes in {', '.join(self.user_profile.skills[:3])}. Respond professionally and helpfully to recruitment-related queries."},
                    {"role": "user", "content": f"{example['instruction']}: {example['input']}"},
                    {"role": "assistant", "content": example["output"]}
                ]
            }
            conversational_data.append(conversation)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversational_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversational format saved to {output_path}")
    
    def preview_dataset(self, num_examples: int = 3):
        """Preview some examples from the dataset."""
        
        print(f"\n📊 Dataset Preview ({num_examples} examples):")
        print("=" * 80)
        
        for i, example in enumerate(self.dataset[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {example['instruction']}")
            print(f"Input: {example['input'][:100]}...")
            print(f"Output: {example['output'][:150]}...")
            print("-" * 40)

def main():
    """Main function to build and save the dataset."""
    
    # Create user profile - CUSTOMIZE THIS WITH YOUR INFORMATION
    user_profile = UserProfile(
        name="Moses Omondi",  # Your actual name
        degree="Master's Degree", 
        university="Missouri State University",
        skills=[
            "Python", "JavaScript", "React", "Node.js", "AWS", 
            "Machine Learning", "Deep Learning", "System Design",
            "API Development", "Database Design", "DevOps", "MLOps"
        ],
        experience_years=5,  # Adjust to your experience
        current_role="Software Engineer",
        linkedin_url="linkedin.com/in/moses-omondi",  # Update with your LinkedIn
        github_url="github.com/moses-omondi",  # Update with your GitHub  
        location="United States",
        industry_focus=[
            "Software Development", "Machine Learning", 
            "Artificial Intelligence", "Cloud Computing"
        ]
    )
    
    # Build dataset
    builder = DatasetBuilder(user_profile)
    dataset = builder.build_dataset()
    
    # Preview dataset
    builder.preview_dataset()
    
    # Save dataset
    builder.save_dataset()
    
    print(f"\n🎉 Dataset creation complete!")
    print(f"📊 Total examples: {len(dataset)}")
    print(f"📁 Saved to: data/training/")
    print(f"🚀 Ready for fine-tuning!")

if __name__ == "__main__":
    main()
