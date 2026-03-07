import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
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
        
        print(f"Top {top_n} contributors for {repo_owner}/{repo_name}:")
        for idx, contributor in enumerate(top_contributors, 1):
            print(f"{idx}. {contributor['login']}: {contributor['contributions']} contributions")
        
        return top_contributors
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    get_top_contributors(owner, repo)import requests

def fetch_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code == 200:
        contributors = response.json()
        for contributor in contributors:
            print(f"Username: {contributor['login']}, Contributions: {contributor['contributions']}")
    else:
        print(f"Failed to fetch contributors. Status code: {response.status_code}")

if __name__ == "__main__":
    fetch_contributors("torvalds", "linux")
import requests
import sys

def get_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code == 200:
        contributors = response.json()
        return contributors
    else:
        print(f"Error: Unable to fetch contributors. Status code: {response.status_code}")
        return None

def display_contributors(contributors):
    if contributors:
        print("Contributors:")
        for contributor in contributors:
            print(f"- {contributor['login']}: {contributor['contributions']} contributions")
    else:
        print("No contributors found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = get_contributors(repo_owner, repo_name)
    display_contributors(contributors)import requests
import csv
import sys

def fetch_contributors(repo_owner, repo_name):
    """
    Fetch the list of contributors for a given GitHub repository.
    Returns a list of dictionaries containing contributor data.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: Unable to fetch contributors. Status code: {response.status_code}")
        return []
    return response.json()

def save_to_csv(contributors, filename="contributors.csv"):
    """
    Save the list of contributors to a CSV file.
    """
    if not contributors:
        print("No contributors to save.")
        return
    fieldnames = ["login", "id", "contributions", "html_url"]
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for contributor in contributors:
                writer.writerow({
                    "login": contributor.get("login", ""),
                    "id": contributor.get("id", ""),
                    "contributions": contributor.get("contributions", 0),
                    "html_url": contributor.get("html_url", "")
                })
        print(f"Contributors saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    print(f"Fetching contributors for {repo_owner}/{repo_name}...")
    contributors = fetch_contributors(repo_owner, repo_name)
    if contributors:
        save_to_csv(contributors)

if __name__ == "__main__":
    main()