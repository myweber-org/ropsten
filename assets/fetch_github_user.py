import requests
import sys

def fetch_github_user(username):
    """Fetch public details of a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user {username}: {e}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_info = fetch_github_user(username)
    
    if user_info:
        print(f"GitHub User: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Profile URL: {user_info['html_url']}")
    else:
        print(f"Could not retrieve information for user '{username}'.")