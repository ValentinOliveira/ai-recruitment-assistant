#!/usr/bin/env python3
"""
Sample Training Data Generator for AI Recruitment Assistant
===========================================================

Generates high-quality training data in Alpaca format for fine-tuning
the recruitment assistant model. Creates diverse examples covering:
- Email responses to candidates
- Interview scheduling
- Rejection letters
- Job descriptions
- Candidate assessment summaries
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse

class RecruitmentDataGenerator:
    """Generates sample training data for recruitment assistant."""
    
    def __init__(self):
        self.candidate_names = [
            "Sarah Johnson", "Michael Chen", "Emily Rodriguez", "David Park",
            "Jessica Williams", "Robert Thompson", "Maria Garcia", "James Wilson",
            "Lisa Anderson", "Christopher Lee", "Amanda Taylor", "Kevin Zhang",
            "Rachel Brown", "Daniel Martinez", "Jennifer Davis", "Matthew Kim"
        ]
        
        self.job_titles = [
            "Software Engineer", "Data Scientist", "Product Manager", "UX Designer",
            "Frontend Developer", "Backend Developer", "DevOps Engineer", "QA Engineer",
            "Marketing Specialist", "Sales Representative", "Business Analyst", "Project Manager",
            "Full Stack Developer", "Mobile Developer", "Security Engineer", "Technical Writer"
        ]
        
        self.companies = [
            "TechCorp", "InnovateAI", "DataSolutions Inc.", "CloudFirst Technologies",
            "NextGen Software", "Digital Dynamics", "FutureTech Labs", "AgileWorks",
            "SmartSystems", "CodeCraft Solutions", "TechPioneer", "InnovateLab"
        ]
        
        self.skills = [
            ["Python", "Machine Learning", "TensorFlow"], ["JavaScript", "React", "Node.js"],
            ["Java", "Spring Boot", "Microservices"], ["Python", "Django", "PostgreSQL"],
            ["C#", ".NET", "Azure"], ["Go", "Kubernetes", "Docker"],
            ["Swift", "iOS", "Mobile Development"], ["Kotlin", "Android", "Firebase"],
            ["AWS", "DevOps", "CI/CD"], ["Data Analysis", "SQL", "Tableau"]
        ]
    
    def generate_interview_scheduling_data(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate interview scheduling examples."""
        examples = []
        
        templates = [
            {
                "instruction": "Schedule an interview with a candidate who applied for a {job_title} position.",
                "input": "Candidate: {name}\nPosition: {job_title}\nAvailable times: {times}\nInterview type: {type}",
                "output": "Dear {name},\n\nThank you for your interest in the {job_title} position at {company}. We were impressed with your application and would like to invite you for an interview.\n\nWe have availability on {interview_time}. The interview will be conducted {interview_format} and is expected to last approximately {duration}.\n\nPlease confirm your availability by replying to this email. We will send you the meeting details once confirmed.\n\nWe look forward to speaking with you!\n\nBest regards,\n{recruiter}"
            },
            {
                "instruction": "Confirm interview details with a candidate.",
                "input": "Candidate confirmed availability for {job_title} interview on Tuesday at 2 PM",
                "output": "Dear {name},\n\nGreat! Your interview for the {job_title} position is confirmed for Tuesday at 2:00 PM.\n\n📅 Interview Details:\n• Date & Time: Tuesday at 2:00 PM\n• Duration: {duration}\n• Format: {interview_format}\n• Interviewers: {interviewers}\n\n{additional_info}\n\nPlease let us know if you have any questions. We're excited to meet you!\n\nBest regards,\n{recruiter}"
            }
        ]
        
        for i in range(count):
            template = random.choice(templates)
            name = random.choice(self.candidate_names)
            job_title = random.choice(self.job_titles)
            company = random.choice(self.companies)
            
            times = random.choice([
                "Monday-Wednesday 2-4 PM", "Tuesday/Thursday mornings", 
                "Wednesday-Friday afternoons", "Next week any time"
            ])
            
            interview_type = random.choice(["Technical", "Behavioral", "Panel", "Phone"])
            interview_format = random.choice(["via Zoom", "in-person at our office", "over the phone"])
            duration = random.choice(["45 minutes", "1 hour", "30 minutes"])
            
            example = {
                "instruction": template["instruction"].format(job_title=job_title),
                "input": template["input"].format(
                    name=name, job_title=job_title, times=times, type=interview_type
                ),
                "output": template["output"].format(
                    name=name, job_title=job_title, company=company,
                    interview_time=random.choice([
                        "Tuesday, March 15th at 2:00 PM",
                        "Thursday, March 17th at 10:30 AM",
                        "Friday, March 18th at 3:00 PM"
                    ]),
                    interview_format=interview_format,
                    duration=duration,
                    date=random.choice(["March 15th", "March 17th", "March 18th"]),
                    time=random.choice(["10:00 AM", "2:00 PM", "3:30 PM"]),
                    interviewers=random.choice([
                        "Sarah (Technical Lead) and Mike (HR)",
                        "The development team",
                        "Jennifer (Hiring Manager)"
                    ]),
                    additional_info=random.choice([
                        "Please bring a copy of your portfolio/recent projects to discuss.",
                        "We'll send you a technical challenge 24 hours before the interview.",
                        "No preparation needed - we'll walk through your experience together."
                    ]),
                    recruiter="Recruitment Team"
                )
            }
            examples.append(example)
        
        return examples
    
    def generate_candidate_response_data(self, count: int = 25) -> List[Dict[str, Any]]:
        """Generate candidate response examples."""
        examples = []
        
        scenarios = [
            {
                "instruction": "Respond to a candidate who inquired about the status of their application.",
                "input": "Candidate {name} applied for {job_title} position 2 weeks ago and is asking for an update",
                "output": "Dear {name},\n\nThank you for following up on your application for the {job_title} position.\n\nI wanted to update you that we have received your application and it is currently being reviewed by our hiring team. We have received a high volume of applications for this role, and we are carefully evaluating each candidate.\n\nWe expect to complete the initial screening process within the next week and will reach out to qualified candidates to schedule interviews.\n\nWe appreciate your patience and continued interest in joining {company}. If you have any questions, please don't hesitate to reach out.\n\nBest regards,\nThe Recruitment Team"
            },
            {
                "instruction": "Send a rejection email to a candidate after the interview process.",
                "input": "Candidate {name} interviewed for {job_title} but was not selected",
                "output": "Dear {name},\n\nThank you for taking the time to interview with us for the {job_title} position. We enjoyed learning about your background and experience.\n\nAfter careful consideration, we have decided to move forward with another candidate whose experience more closely aligns with our current needs.\n\nThis was a difficult decision as we were impressed with your {positive_feedback}. We encourage you to apply for future opportunities that match your skills and interests.\n\nWe wish you the best of luck in your job search and thank you again for your interest in {company}.\n\nBest regards,\n{recruiter}"
            },
            {
                "instruction": "Send a job offer to a successful candidate.",
                "input": "Candidate {name} has been selected for the {job_title} position with salary ${salary}k",
                "output": "Dear {name},\n\nCongratulations! We are delighted to extend you an offer for the {job_title} position at {company}.\n\n🎉 Offer Details:\n• Position: {job_title}\n• Starting Salary: ${salary},000 per year\n• Start Date: {start_date}\n• Benefits: Health insurance, 401(k) matching, flexible PTO, remote work options\n• Reporting to: {manager}\n\nThis offer is contingent upon successful completion of our background check process.\n\nPlease review the attached formal offer letter and let us know your decision by {deadline}. We're excited about the possibility of you joining our team!\n\nIf you have any questions, please don't hesitate to reach out.\n\nWelcome to the team!\n\nBest regards,\n{recruiter}"
            }
        ]
        
        for i in range(count):
            scenario = random.choice(scenarios)
            name = random.choice(self.candidate_names)
            job_title = random.choice(self.job_titles)
            company = random.choice(self.companies)
            
            # Handle different scenario inputs
            if "salary" in scenario["input"]:
                input_text = scenario["input"].format(name=name, job_title=job_title, salary=random.choice([75, 85, 95, 105, 115, 125]))
            else:
                input_text = scenario["input"].format(name=name, job_title=job_title)
                
            example = {
                "instruction": scenario["instruction"],
                "input": input_text,
                "output": scenario["output"].format(
                    name=name,
                    job_title=job_title,
                    company=company,
                    positive_feedback=random.choice([
                        "technical skills and enthusiasm",
                        "communication abilities and relevant experience",
                        "problem-solving approach and cultural fit",
                        "portfolio and collaborative mindset"
                    ]),
                    salary=random.choice([75, 85, 95, 105, 115, 125]),
                    start_date=random.choice([
                        "April 1st, 2024", "March 15th, 2024", "April 15th, 2024"
                    ]),
                    manager=random.choice([
                        "Sarah Johnson (Engineering Manager)",
                        "Mike Chen (Team Lead)",
                        "Jennifer Davis (Product Manager)"
                    ]),
                    deadline=random.choice([
                        "Friday, March 10th", "Monday, March 13th", "Wednesday, March 8th"
                    ]),
                    recruiter="The Recruitment Team"
                )
            }
            examples.append(example)
        
        return examples
    
    def generate_job_description_data(self, count: int = 15) -> List[Dict[str, Any]]:
        """Generate job description examples."""
        examples = []
        
        for i in range(count):
            job_title = random.choice(self.job_titles)
            company = random.choice(self.companies)
            skills_list = random.choice(self.skills)
            
            example = {
                "instruction": f"Create a job description for a {job_title} position.",
                "input": f"Company: {company}\nPosition: {job_title}\nKey skills: {', '.join(skills_list)}",
                "output": f"# {job_title} - {company}\n\n## About the Role\nWe are seeking a talented {job_title} to join our dynamic team at {company}. This is an exciting opportunity to work on cutting-edge projects and contribute to our mission of innovation.\n\n## Key Responsibilities\n• Develop and maintain high-quality software solutions\n• Collaborate with cross-functional teams to deliver projects\n• Participate in code reviews and technical discussions\n• Contribute to architectural decisions and best practices\n• Mentor junior team members and share knowledge\n\n## Required Skills\n• {skills_list[0]} - 3+ years of experience\n• {skills_list[1]} - Strong proficiency\n• {skills_list[2]} - Hands-on experience\n• Strong problem-solving and communication skills\n• Experience with agile development methodologies\n\n## What We Offer\n• Competitive salary and equity package\n• Comprehensive health and wellness benefits\n• Flexible work arrangements and PTO\n• Professional development opportunities\n• Collaborative and inclusive work environment\n\nReady to make an impact? Apply now!"
            }
            examples.append(example)
        
        return examples
    
    def generate_candidate_assessment_data(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate candidate assessment examples."""
        examples = []
        
        for i in range(count):
            name = random.choice(self.candidate_names)
            job_title = random.choice(self.job_titles)
            
            assessment_type = random.choice([
                "technical interview", "behavioral interview", "coding challenge", "portfolio review"
            ])
            
            strengths = random.sample([
                "Strong technical skills", "Excellent communication", "Problem-solving ability",
                "Cultural fit", "Leadership potential", "Collaborative mindset",
                "Attention to detail", "Learning agility", "Industry experience"
            ], 3)
            
            areas_for_growth = random.sample([
                "Framework-specific knowledge", "System design experience", "Team leadership",
                "Public speaking", "Project management", "Cross-team collaboration"
            ], 2)
            
            example = {
                "instruction": f"Write a candidate assessment summary after a {assessment_type}.",
                "input": f"Candidate: {name}\nPosition: {job_title}\nAssessment type: {assessment_type}",
                "output": f"# Candidate Assessment: {name}\n\n**Position:** {job_title}\n**Assessment Date:** {random.choice(['March 10, 2024', 'March 11, 2024', 'March 12, 2024'])}\n**Assessment Type:** {assessment_type.title()}\n\n## Summary\n{name} demonstrated {random.choice(['strong', 'solid', 'excellent'])} performance during the {assessment_type}. They showed good understanding of the role requirements and expressed genuine interest in the position.\n\n## Strengths\n• {strengths[0]}\n• {strengths[1]}\n• {strengths[2]}\n\n## Areas for Growth\n• {areas_for_growth[0]}\n• {areas_for_growth[1]}\n\n## Technical Assessment\n{random.choice(['Passed', 'Strong performance', 'Met expectations'])} - {random.choice(['Solved problems efficiently', 'Demonstrated solid coding practices', 'Showed good architectural thinking'])}\n\n## Recommendation\n{random.choice(['Recommend for next round', 'Strong hire recommendation', 'Proceed with reference checks'])}\n\n## Next Steps\n{random.choice(['Schedule final interview with hiring manager', 'Conduct technical deep-dive session', 'Prepare offer package'])}\n\n**Interviewer:** {random.choice(['Sarah Johnson', 'Mike Chen', 'Jennifer Davis'])}"
            }
            examples.append(example)
        
        return examples
    
    def generate_all_data(self, total_examples: int = 100) -> List[Dict[str, Any]]:
        """Generate all types of training data."""
        # Distribute examples across different categories
        interview_count = int(total_examples * 0.25)  # 25%
        response_count = int(total_examples * 0.35)   # 35%
        job_desc_count = int(total_examples * 0.2)    # 20%
        assessment_count = total_examples - (interview_count + response_count + job_desc_count)  # 20%
        
        all_examples = []
        all_examples.extend(self.generate_interview_scheduling_data(interview_count))
        all_examples.extend(self.generate_candidate_response_data(response_count))
        all_examples.extend(self.generate_job_description_data(job_desc_count))
        all_examples.extend(self.generate_candidate_assessment_data(assessment_count))
        
        # Shuffle to mix different types
        random.shuffle(all_examples)
        
        return all_examples

def main():
    """Main function to generate training data."""
    parser = argparse.ArgumentParser(description="Generate sample training data for AI Recruitment Assistant")
    parser.add_argument("--output", "-o", type=str, default="data/training/alpaca_format.json",
                       help="Output file path (default: data/training/alpaca_format.json)")
    parser.add_argument("--count", "-c", type=int, default=100,
                       help="Number of training examples to generate (default: 100)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducible generation (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Generating {args.count} training examples...")
    
    # Generate data
    generator = RecruitmentDataGenerator()
    training_data = generator.generate_all_data(args.count)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(training_data)} examples")
    print(f"💾 Saved to: {output_path}")
    
    # Show sample data
    print("\n📊 Sample Examples:")
    for i, example in enumerate(training_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example['instruction'][:80]}...")
        print(f"Input: {example['input'][:80]}...")
        print(f"Output: {example['output'][:100]}...")
    
    print(f"\n🎯 Data Distribution:")
    instruction_types = {}
    for example in training_data:
        key = example['instruction'].split()[0:3]  # First 3 words
        key = ' '.join(key)
        instruction_types[key] = instruction_types.get(key, 0) + 1
    
    for inst_type, count in sorted(instruction_types.items()):
        print(f"  • {inst_type}: {count} examples")

if __name__ == "__main__":
    main()
