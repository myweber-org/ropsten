import requests

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_list = [repo['name'] for repo in repos]
        return repo_list
    else:
        return f"Error: Unable to fetch repositories (Status code: {response.status_code})"

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    repositories = get_github_repos(user)
    
    if isinstance(repositories, list):
        print(f"Public repositories for {user}:")
        for repo in repositories:
            print(f" - {repo}")
    else:
        print(repositories)import requests
import sys

def fetch_user_repos(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"page": page, "per_page": per_page}
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return

    for repo in repos:
        name = repo.get("name", "N/A")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        print(f"Name: {name}")
        print(f"Description: {description}")
        print(f"Stars: {stars} | Forks: {forks}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)

    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    repos = fetch_user_repos(username, page, per_page)
    if repos is not None:
        display_repos(repos)

if __name__ == "__main__":
    main()import requests
import sys

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        if repos:
            print(f"Public repositories for {username}:")
            for repo in repos:
                print(f"- {repo['name']}: {repo['html_url']}")
        else:
            print(f"No public repositories found for {username}.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_github_repos(username)
import requests
import sys

def fetch_github_repos(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'type': 'owner',
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Python-Script'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repos(repos):
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
        print(f"  Last Updated: {updated}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    print(f"Fetching repositories for user: {username} (Page {page}, {per_page} per page)")
    print("=" * 50)
    
    repos = fetch_github_repos(username, page, per_page)
    display_repos(repos)

if __name__ == "__main__":
    main()