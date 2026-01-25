import requests

def fetch_github_repos(username, per_page=30, page=1):
    """Fetch repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    params = {"per_page": per_page, "page": page}
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []

def display_repos(repos):
    """Display repository information."""
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        name = repo.get("name", "N/A")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        language = repo.get("language", "Not specified")
        
        print(f"Name: {name}")
        print(f"Description: {description}")
        print(f"Stars: {stars} | Forks: {forks} | Language: {language}")
        print("-" * 50)

def main():
    username = input("Enter GitHub username: ")
    page = 1
    
    while True:
        repos = fetch_github_repos(username, page=page)
        display_repos(repos)
        
        if len(repos) < 30:
            print("No more repositories.")
            break
        
        choice = input("Fetch next page? (y/n): ").lower()
        if choice != 'y':
            break
        page += 1

if __name__ == "__main__":
    main()