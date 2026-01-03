import requests
import sys

def fetch_repositories(username):
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
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    fetch_repositories(username)import requests
import sys

def fetch_github_repos(username, token=None, per_page=100):
    """
    Fetch all repositories for a given GitHub username.
    Uses pagination to retrieve all repos.
    """
    repos = []
    page = 1
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    while True:
        url = f'https://api.github.com/users/{username}/repos'
        params = {'page': page, 'per_page': per_page}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            repos.extend(data)
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repositories: {e}", file=sys.stderr)
            break

    return repos

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [token]")
        sys.exit(1)

    username = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    repos = fetch_github_repos(username, token)

    if repos:
        print(f"Found {len(repos)} repositories for user '{username}':")
        for repo in repos:
            print(f"- {repo['name']}: {repo['html_url']}")
    else:
        print(f"No repositories found for user '{username}'.")

if __name__ == '__main__':
    main()