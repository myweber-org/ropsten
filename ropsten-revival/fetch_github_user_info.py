import requests
import json

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    else:
        return None

def display_user_info(user_info):
    if user_info:
        print(f"Name: {user_info['name']}")
        print(f"Bio: {user_info['bio']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
    else:
        print("User not found or API request failed.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user_info(username)
    display_user_info(user_info)import requests
import sys

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'login': user_data.get('login'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return None
        else:
            print(f"HTTP error occurred: {e}", file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <github_username>")
        sys.exit(1)

    username = sys.argv[1]
    user_info = get_github_user_info(username)

    if user_info is None:
        print(f"User '{username}' not found or error occurred.")
        sys.exit(1)

    print(f"GitHub User: {user_info['login']}")
    print(f"Name: {user_info['name']}")
    print(f"Public Repositories: {user_info['public_repos']}")
    print(f"Followers: {user_info['followers']}")
    print(f"Following: {user_info['following']}")
    print(f"Account Created: {user_info['created_at']}")

if __name__ == "__main__":
    main()