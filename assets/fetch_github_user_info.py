import requests

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
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
            'created_at': user_data.get('created_at')
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        info = get_github_user_info(username)
        if info:
            print(f"\nGitHub User: {info['login']}")
            print(f"Name: {info['name']}")
            print(f"Public Repositories: {info['public_repos']}")
            print(f"Followers: {info['followers']}")
            print(f"Following: {info['following']}")
            print(f"Account Created: {info['created_at']}")
        else:
            print("Could not retrieve user information.")
    else:
        print("No username provided.")import requests
import sys

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        if response.status_code == 404:
            print(f"Error: User '{username}' not found.")
        else:
            print(f"HTTP error occurred: {err}")
        sys.exit(1)
    except requests.exceptions.RequestException as err:
        print(f"Request error occurred: {err}")
        sys.exit(1)

def display_user_info(user_data):
    """Display selected user information."""
    if not user_data:
        return

    print(f"GitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    user_data = fetch_github_user(username)
    display_user_info(user_data)import requests
import json

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def display_user_info(user_data):
    """
    Display selected user information in a readable format.
    """
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Bio: {user_data.get('bio', 'Not provided')}")
        print(f"Public Repositories: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("No user data to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        data = get_github_user_info(username)
        display_user_info(data)
    else:
        print("No username provided.")import requests

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        # Extract and return selected fields
        info = {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "company": user_data.get("company"),
            "blog": user_data.get("blog"),
            "location": user_data.get("location"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "created_at": user_data.get("created_at")
        }
        return info
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def display_user_info(user_info):
    """
    Pretty print the user information.
    """
    if not user_info:
        print("No user information to display.")
        return
    
    print("GitHub User Information:")
    print("-" * 30)
    for key, value in user_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    # Example usage
    username = input("Enter a GitHub username: ").strip()
    if username:
        info = get_github_user_info(username)
        display_user_info(info)
    else:
        print("No username provided.")