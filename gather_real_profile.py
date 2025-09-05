#!/usr/bin/env python3
"""
Gather Moses Omondi's Real Profile Information
==============================================

Extract real information from GitHub API and create accurate training dataset.
"""

import requests
import json
import time
from datetime import datetime

def gather_github_profile():
    """Gather Moses's GitHub profile information."""
    print("🔍 Gathering Moses Omondi's GitHub profile information...")
    
    try:
        # Get user profile
        response = requests.get("https://api.github.com/users/Moses-Omondi")
        if response.status_code == 200:
            profile = response.json()
            
            print(f"✅ GitHub Profile Found:")
            print(f"   Name: {profile.get('name', 'Moses Omondi')}")
            print(f"   Bio: {profile.get('bio', 'All things code.')}")
            print(f"   Location: {profile.get('location', 'Not specified')}")
            print(f"   Company: {profile.get('company', 'Not specified')}")
            print(f"   Public Repos: {profile.get('public_repos', 0)}")
            print(f"   Followers: {profile.get('followers', 0)}")
            print(f"   Following: {profile.get('following', 0)}")
            print(f"   Created: {profile.get('created_at', 'Unknown')}")
            
            # Get repositories
            repos_response = requests.get(f"https://api.github.com/users/Moses-Omondi/repos?sort=updated&per_page=10")
            repos = []
            if repos_response.status_code == 200:
                repos_data = repos_response.json()
                for repo in repos_data[:10]:  # Top 10 recent repos
                    repos.append({
                        'name': repo.get('name'),
                        'description': repo.get('description'),
                        'language': repo.get('language'),
                        'stars': repo.get('stargazers_count', 0),
                        'updated': repo.get('updated_at'),
                        'topics': repo.get('topics', [])
                    })
                
                print(f"✅ Found {len(repos)} recent repositories")
                
            return {
                'profile': profile,
                'repositories': repos,
                'gathered_at': datetime.now().isoformat()
            }
        else:
            print(f"❌ Failed to fetch GitHub profile: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error gathering GitHub information: {e}")
        return None

def create_real_dataset(github_data):
    """Create training dataset based on real GitHub information."""
    
    profile = github_data['profile']
    repos = github_data['repositories']
    
    # Extract technologies from repositories
    languages = set()
    topics = set()
    for repo in repos:
        if repo['language']:
            languages.add(repo['language'])
        topics.update(repo['topics'])
    
    # Create real dataset based on actual profile
    real_dataset = [
        {
            "instruction": "You are Moses Omondi's personal AI recruitment assistant. Provide information about Moses's real GitHub profile and development experience.",
            "input": "Tell me about Moses Omondi's GitHub profile and his development background.",
            "output": f"Moses Omondi is an active software developer with a strong presence on GitHub.\n\n**GitHub Profile Overview:**\n- **Profile**: @Moses-Omondi on GitHub\n- **Bio**: {profile.get('bio', 'All things code.')}\n- **Public Repositories**: {profile.get('public_repos', 0)} repositories\n- **Community**: {profile.get('followers', 0)} followers, following {profile.get('following', 0)} developers\n- **Member Since**: {profile.get('created_at', 'Unknown')}\n\n**Technical Expertise (Based on Recent Repositories):**\n- **Primary Languages**: {', '.join(sorted(languages)) if languages else 'Multiple languages'}\n- **Technology Focus**: {', '.join(sorted(topics)) if topics else 'Full-stack development'}\n\n**Development Activity:**\nMoses maintains an active development profile with {len(repos)} recent projects showcasing his technical versatility and commitment to coding excellence.\n\nHis GitHub profile demonstrates practical hands-on experience with modern development tools and technologies. You can view his complete portfolio at https://github.com/Moses-Omondi\n\nWould you like me to elaborate on any specific aspects of his technical background or discuss how his development experience aligns with your role requirements?"
        },
        {
            "instruction": "You are Moses Omondi's personal AI recruitment assistant. Discuss Moses's specific programming languages and technical skills based on his actual repositories.",
            "input": "What programming languages and technologies does Moses Omondi work with based on his GitHub activity?",
            "output": create_skills_output(profile, repos, languages, topics)
        }
    ]
    
    return real_dataset

def main():
    print("🚀 Gathering Moses Omondi's Real Profile Information")
    print("=" * 60)
    
    # Gather GitHub data
    github_data = gather_github_profile()
    
    if github_data:
        # Create real dataset
        real_dataset = create_real_dataset(github_data)
        
        # Save the real profile data
        with open('data/moses_real_profile.json', 'w', encoding='utf-8') as f:
            json.dump(github_data, f, indent=2, ensure_ascii=False)
        
        print("✅ Real profile data saved to data/moses_real_profile.json")
        
        # Save the real dataset
        with open('data/moses_real_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(real_dataset, f, indent=2, ensure_ascii=False)
        
        print("✅ Real dataset saved to data/moses_real_dataset.json")
        print(f"📊 Created {len(real_dataset)} training examples based on real profile")
        
        # Show a summary
        print("\n📋 Real Profile Summary:")
        profile = github_data['profile']
        print(f"   • Name: {profile.get('name', 'Moses Omondi')}")
        print(f"   • Repositories: {profile.get('public_repos', 0)}")
        print(f"   • Followers: {profile.get('followers', 0)}")
        if github_data['repositories']:
            languages = set(repo['language'] for repo in github_data['repositories'] if repo['language'])
            if languages:
                print(f"   • Languages: {', '.join(sorted(languages))}")
        
    else:
        print("❌ Failed to gather real profile information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
