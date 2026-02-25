
import requests

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'login': user_data.get('login'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    else:
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user_info(username)
    
    if user_info:
        print(f"Name: {user_info['name']}")
        print(f"Username: {user_info['login']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")
    else:
        print(f"User '{username}' not found or API error occurred.")
import requests
import time

def fetch_github_user(username, token=None):
    """
    Fetch public information for a GitHub user.
    Handles rate limiting by waiting and retrying.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403 and 'rate limit' in response.text.lower():
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_duration = max(reset_time - time.time(), 0) + 5
            print(f"Rate limited. Waiting {sleep_duration:.0f} seconds.")
            time.sleep(sleep_duration)
            continue
        else:
            response.raise_for_status()

if __name__ == "__main__":
    try:
        user_data = fetch_github_user("octocat")
        print(f"User: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public repos: {user_data.get('public_repos')}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user data: {e}")import requests

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
            'following': user_data.get('following')
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    details = fetch_github_user(username)
    if details:
        print(f"User: {details['login']}")
        print(f"Name: {details['name']}")
        print(f"Public Repos: {details['public_repos']}")
        print(f"Followers: {details['followers']}")
        print(f"Following: {details['following']}")import requests

def fetch_github_user(username):
    """
    Fetches public details of a GitHub user.
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
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        details = fetch_github_user(username)
        if details:
            print(f"User: {details['login']}")
            print(f"Name: {details['name']}")
            print(f"Public Repos: {details['public_repos']}")
            print(f"Followers: {details['followers']}")
            print(f"Following: {details['following']}")
            print(f"Account Created: {details['created_at']}")
        else:
            print("Could not fetch user details.")
import requests
import sys

def fetch_github_user(username):
    """
    Fetches public information for a given GitHub username.
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
    Displays selected user information in a formatted way.
    """
    if not user_data:
        print("No user data to display.")
        return

    print(f"GitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    data = fetch_github_user(username)

    if data:
        display_user_info(data)
    else:
        print(f"Failed to fetch data for user '{username}'.")