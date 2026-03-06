import requests
import sys

def fetch_user_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        return repos
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    
    print(f"Found {len(repos)} public repositories:")
    for repo in repos:
        print(f"- {repo['name']}: {repo['description'] or 'No description'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_user_repos(username)
    
    if repos:
        display_repos(repos)