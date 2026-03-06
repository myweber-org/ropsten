
import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
    """
    Fetch top contributors for a GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contributors = response.json()
        
        if not contributors:
            print("No contributors found.")
            return []
        
        sorted_contributors = sorted(contributors, key=lambda x: x.get('contributions', 0), reverse=True)
        top_contributors = sorted_contributors[:top_n]
        
        return top_contributors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return []

def display_contributors(contributors):
    """
    Display contributor information.
    """
    if not contributors:
        return
    
    print("Top Contributors:")
    for idx, contributor in enumerate(contributors, start=1):
        login = contributor.get('login', 'N/A')
        contributions = contributor.get('contributions', 0)
        print(f"{idx}. {login}: {contributions} contributions")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name> [top_n]")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    top_contributors = get_top_contributors(repo_owner, repo_name, top_n)
    display_contributors(top_contributors)import requests
import sys

def get_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch contributors. Status code: {response.status_code}")
        return []
    
    contributors = response.json()
    return contributors

def display_contributors(contributors):
    if not contributors:
        print("No contributors found.")
        return
    
    print("Contributors:")
    for contributor in contributors:
        print(f"- {contributor['login']}: {contributor['contributions']} contributions")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = get_contributors(repo_owner, repo_name)
    display_contributors(contributors)import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
    """
    Fetch top contributors for a GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contributors = response.json()
        
        if not contributors:
            print(f"No contributors found for {repo_owner}/{repo_name}")
            return []
        
        sorted_contributors = sorted(contributors, key=lambda x: x.get('contributions', 0), reverse=True)
        top_contributors = sorted_contributors[:top_n]
        
        return top_contributors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching contributors: {e}")
        return []

def display_contributors(contributors):
    """
    Display contributor information.
    """
    if not contributors:
        return
    
    print("Top Contributors:")
    for idx, contributor in enumerate(contributors, 1):
        username = contributor.get('login', 'N/A')
        contributions = contributor.get('contributions', 0)
        print(f"{idx}. {username}: {contributions} contributions")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    
    top_contributors = get_top_contributors(owner, repo)
    display_contributors(top_contributors)