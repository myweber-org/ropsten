import requests
import sys

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'name': data.get('name'),
            'company': data.get('company'),
            'blog': data.get('blog'),
            'location': data.get('location'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'following': data.get('following')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

def display_user_info(username, info):
    """Display the fetched user information."""
    if info is None:
        print(f"Could not retrieve information for user '{username}'.")
        return

    print(f"GitHub User: {username}")
    print("-" * 30)
    print(f"Name: {info['name'] or 'Not provided'}")
    print(f"Company: {info['company'] or 'Not provided'}")
    print(f"Blog/Website: {info['blog'] or 'Not provided'}")
    print(f"Location: {info['location'] or 'Not provided'}")
    print(f"Public Repositories: {info['public_repos']}")
    print(f"Followers: {info['followers']}")
    print(f"Following: {info['following']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_data.py <github_username>")
        sys.exit(1)

    target_username = sys.argv[1]
    user_data = fetch_github_user(target_username)
    display_user_info(target_username, user_data)