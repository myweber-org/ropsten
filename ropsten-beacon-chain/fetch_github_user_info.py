import requests

def get_github_user_info(username):
    """
    Fetches public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()

        # Extract and return specific fields
        info = {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
        return info
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        result = get_github_user_info(username)
        if result:
            print(f"User: {result['login']}")
            print(f"Name: {result['name']}")
            print(f"Public Repositories: {result['public_repos']}")
            print(f"Followers: {result['followers']}")
            print(f"Following: {result['following']}")
            print(f"Profile URL: {result['html_url']}")
        else:
            print("Failed to fetch user information.")
    else:
        print("No username provided.")