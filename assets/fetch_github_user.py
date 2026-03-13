import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch user. Status code: {response.status_code}"}

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    print(user_data)import requests
import json

def fetch_github_user(username):
    """
    Fetches public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Python-Script'
    }
    try:
        response = requests.get(url, headers=headers)
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
    Prints selected user information in a readable format.
    """
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("No user data to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        data = fetch_github_user(username)
        display_user_info(data)
    else:
        print("No username provided.")import requests

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
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def display_user_info(user_info):
    """
    Display the fetched user information in a formatted way.
    """
    if user_info:
        print(f"GitHub User: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Profile URL: {user_info['html_url']}")
    else:
        print("No user information to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        info = get_github_user_info(username)
        display_user_info(info)
    else:
        print("No username provided.")
import requests

def fetch_github_user(username):
    """
    Fetch public information for a given GitHub username.
    Returns a dictionary with user details or None if not found.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user data: {e}")
        return None

def display_user_info(user_data):
    """Display key user information in a formatted way."""
    if not user_data:
        print("No user data to display.")
        return
    
    print(f"GitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = fetch_github_user(username)
        display_user_info(user_info)
    else:
        print("No username entered.")import requests

def fetch_github_user(username):
    """
    Fetches public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

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
            "html_url": user_data.get("html_url")
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        result = fetch_github_user(username)
        if result:
            print(f"User: {result['login']}")
            print(f"Name: {result['name']}")
            print(f"Public Repos: {result['public_repos']}")
            print(f"Followers: {result['followers']}")
            print(f"Following: {result['following']}")
            print(f"Profile URL: {result['html_url']}")
        else:
            print("Failed to fetch user data.")
    else:
        print("No username provided.")import requests

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
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def main():
    username = input("Enter a GitHub username: ")
    user_info = get_github_user_info(username)
    if user_info:
        print(f"\nGitHub User: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Profile URL: {user_info['html_url']}")
    else:
        print("Failed to fetch user information.")

if __name__ == "__main__":
    main()
import requests
import sys

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
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
    """Display selected user information."""
    if not user_data:
        print("No user data to display.")
        return

    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Public Repositories: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <github_username>")
        sys.exit(1)

    username = sys.argv[1]
    data = fetch_github_user(username)

    if data:
        display_user_info(data)
    else:
        print(f"Failed to fetch data for user '{username}'.")