import requests

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

if __name__ == "__main__":
    username = input("Enter a GitHub username: ")
    info = get_github_user_info(username)
    if info:
        print(f"Username: {info['login']}")
        print(f"Name: {info['name']}")
        print(f"Public Repositories: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
        print(f"Profile URL: {info['html_url']}")
    else:
        print("Failed to fetch user information.")import requests

def fetch_github_user(username):
    """
    Fetch public profile information for a given GitHub username.
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
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    result = fetch_github_user(username)
    if result:
        print(f"User: {result['login']}")
        print(f"Name: {result['name']}")
        print(f"Public Repositories: {result['public_repos']}")
        print(f"Followers: {result['followers']}")
        print(f"Following: {result['following']}")
        print(f"Profile URL: {result['html_url']}")import requests
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
    print(f"Public Repositories: {user_data.get('public_repos')}")
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