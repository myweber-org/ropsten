import requests
import sys

def fetch_repo_info(username, repo_name):
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repo_data = response.json()
        
        print(f"Repository: {repo_data['full_name']}")
        print(f"Description: {repo_data['description'] or 'No description'}")
        print(f"Stars: {repo_data['stargazers_count']}")
        print(f"Forks: {repo_data['forks_count']}")
        print(f"Open Issues: {repo_data['open_issues_count']}")
        print(f"Language: {repo_data['language'] or 'Not specified'}")
        print(f"URL: {repo_data['html_url']}")
        
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching repository: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_repo_info.py <username> <repository>")
        sys.exit(1)
    
    username = sys.argv[1]
    repo_name = sys.argv[2]
    fetch_repo_info(username, repo_name)