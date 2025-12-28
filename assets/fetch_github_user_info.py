
import requests

def fetch_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            "login": data.get("login"),
            "name": data.get("name"),
            "public_repos": data.get("public_repos"),
            "followers": data.get("followers"),
            "following": data.get("following"),
            "created_at": data.get("created_at")
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    info = fetch_github_user_info(user)
    if info:
        print(f"User: {info['login']}")
        print(f"Name: {info['name']}")
        print(f"Public Repos: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
        print(f"Account Created: {info['created_at']}")