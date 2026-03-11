
import requests

def fetch_github_user(username):
    """
    Fetch basic information for a GitHub user.
    """
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def display_user_info(user_info):
    """
    Display the fetched user information.
    """
    if user_info:
        print(f"Username: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")
    else:
        print("No user information to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    info = fetch_github_user(username)
    display_user_info(info)
import requests

def fetch_github_user(username):
    """Fetch public details of a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "created_at": user_data.get("created_at")
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        result = fetch_github_user(username)
        if result:
            print(f"User: {result['login']}")
            print(f"Name: {result['name']}")
            print(f"Public Repos: {result['public_repos']}")
            print(f"Followers: {result['followers']}")
            print(f"Following: {result['following']}")
            print(f"Account Created: {result['created_at']}")
        else:
            print("Failed to fetch user details.")