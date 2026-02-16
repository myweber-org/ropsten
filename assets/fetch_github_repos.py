import requests
import sys

def fetch_github_repos(username):
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
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    fetch_github_repos(username)
import requests
import argparse
import sys

def get_user_repositories(username, sort_by='created', direction='desc'):
    """
    Fetch public repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {'sort': sort_by, 'direction': direction}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No public repositories found for user: {username}")
            return []
            
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        sys.exit(1)

def display_repositories(repos, max_display=10):
    """
    Display repository information in a formatted way.
    """
    if not repos:
        return
        
    print(f"\nFound {len(repos)} repositories. Displaying first {min(max_display, len(repos))}:\n")
    print(f"{'Name':<30} {'Stars':<10} {'Language':<15} {'Updated':<20}")
    print("-" * 80)
    
    for repo in repos[:max_display]:
        name = repo.get('name', 'N/A')[:28]
        stars = repo.get('stargazers_count', 0)
        language = repo.get('language', 'Not specified')[:13]
        updated = repo.get('updated_at', 'N/A')[:19]
        
        print(f"{name:<30} {stars:<10} {language:<15} {updated:<20}")

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'], 
                       default='created', help='Sort repositories by field')
    parser.add_argument('--direction', choices=['asc', 'desc'], 
                       default='desc', help='Sort direction')
    parser.add_argument('--limit', type=int, default=10, 
                       help='Maximum number of repositories to display')
    
    args = parser.parse_args()
    
    repos = get_user_repositories(args.username, args.sort, args.direction)
    display_repositories(repos, args.limit)

if __name__ == "__main__":
    main()