import requests

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    fetch_github_repos(username)
import requests
import sys

def get_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        return [repo['name'] for repo in repos]
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repositories = get_github_repos(username)
    
    if repositories:
        print(f"Public repositories for '{username}':")
        for repo in repositories:
            print(f"  - {repo}")
    else:
        print(f"No public repositories found for '{username}'.")