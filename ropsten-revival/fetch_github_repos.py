import requests

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_list = [repo['name'] for repo in repos]
        return repo_list
    else:
        return f"Error: Unable to fetch repositories (Status code: {response.status_code})"

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    repositories = get_github_repos(user)
    
    if isinstance(repositories, list):
        print(f"Public repositories for {user}:")
        for repo in repositories:
            print(f" - {repo}")
    else:
        print(repositories)