import requests
import sys

def get_stars(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        stars = data.get('stargazers_count', 'N/A')
        return stars
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_stars.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    star_count = get_stars(owner, repo)
    
    if star_count is not None:
        print(f"The repository {owner}/{repo} has {star_count} stars.")