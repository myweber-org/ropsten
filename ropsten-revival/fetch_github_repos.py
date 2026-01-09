import requests
import json

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        print(f"Public repositories for user '{username}':")
        for repo in repos:
            print(f"- {repo['name']}: {repo['description'] or 'No description'}")
        return repos
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    user = input("Enter a GitHub username: ")
    fetch_github_repos(user)