
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
        print(f"Error fetching repositories: {e}", file=sys.stderr)
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
        
        print(f"📦 {name}")
        print(f"   {description}")
        print(f"   ⭐ {stars} | 🍴 {forks} | 🗣️ {language} | 📅 {updated}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    print(f"Fetching repositories for user: {username} (Page {page}, {per_page} per page)")
    print("=" * 60)
    
    repos = fetch_repositories(username, page, per_page)
    display_repositories(repos)

if __name__ == "__main__":
    main()import requests

def fetch_github_repos(username, per_page=30, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'per_page': per_page,
        'page': page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
        return repos
    else:
        print(f"Failed to fetch repositories: {response.status_code}")
        return None

if __name__ == "__main__":
    fetch_github_repos("octocat")import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"page": page, "per_page": per_page}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description'] or 'No description'}")
        print(f"URL: {repo['html_url']}")
        print(f"Stars: {repo['stargazers_count']}")
        print(f"Forks: {repo['forks_count']}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_repositories(username, page, per_page)
    if repos:
        display_repositories(repos)

if __name__ == "__main__":
    main()