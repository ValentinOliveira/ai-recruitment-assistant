#!/usr/bin/env python3
"""
Simple GitHub Profile Extractor for Moses Omondi
================================================
"""

import requests
import json

def main():
    print("🔍 Gathering Moses Omondi's GitHub profile...")
    
    # Get GitHub profile
    response = requests.get("https://api.github.com/users/Moses-Omondi")
    if response.status_code == 200:
        profile = response.json()
        print(f"✅ Found GitHub profile:")
        print(f"   Name: {profile.get('name')}")
        print(f"   Bio: {profile.get('bio')}")
        print(f"   Company: {profile.get('company')}")
        print(f"   Location: {profile.get('location')}")
        print(f"   Repos: {profile.get('public_repos')}")
        print(f"   Followers: {profile.get('followers')}")
        print(f"   Created: {profile.get('created_at')}")
        
        # Save the profile
        with open('data/github_profile.json', 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
        
        print("✅ Profile saved to data/github_profile.json")
        
        # Get repositories
        repos_response = requests.get("https://api.github.com/users/Moses-Omondi/repos?sort=updated&per_page=20")
        if repos_response.status_code == 200:
            repos = repos_response.json()
            
            print(f"\n📁 Recent repositories ({len(repos)}):")
            languages = set()
            
            for i, repo in enumerate(repos[:10]):
                print(f"   {i+1}. {repo.get('name')}: {repo.get('language', 'N/A')}")
                if repo.get('language'):
                    languages.add(repo.get('language'))
            
            print(f"\n💻 Languages found: {', '.join(sorted(languages))}")
            
            # Save repositories
            with open('data/github_repos.json', 'w', encoding='utf-8') as f:
                json.dump(repos, f, indent=2)
            
            print("✅ Repositories saved to data/github_repos.json")
    
    else:
        print(f"❌ Failed to get profile: {response.status_code}")

if __name__ == "__main__":
    main()
