
import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        language = repo.get('language', 'Not specified')
        updated = repo.get('updated_at', 'N/A')[:10]
        
        print(f"Repository: {name}")
        print(f"  Description: {description}")
        print(f"  Language: {language}")
        print(f"  Stars: {stars} | Forks: {forks}")
        print(f"  Last updated: {updated}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_user_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    print(f"Fetching repositories for user: {username} (Page {page}, {per_page} per page)")
    print("=" * 50)
    
    repos = fetch_repositories(username, page, per_page)
    display_repositories(repos)

if __name__ == "__main__":
    main()import requests

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        if repos:
            print(f"Public repositories for user '{username}':")
            for repo in repos:
                print(f"- {repo['name']}: {repo['description'] or 'No description'}")
        else:
            print(f"No public repositories found for user '{username}'.")
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        fetch_github_repos(username)
    else:
        print("No username provided.")