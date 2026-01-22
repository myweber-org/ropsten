import requests
import sys
from datetime import datetime, timedelta

def fetch_recent_issues(owner, repo, days=7):
    """Fetch issues updated in the last N days."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    params = {
        'state': 'all',
        'since': since_date,
        'sort': 'updated',
        'direction': 'desc',
        'per_page': 20
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return None

def display_issues(issues):
    """Display issue information."""
    if not issues:
        print("No issues found.")
        return
    
    for issue in issues:
        print(f"#{issue['number']}: {issue['title']}")
        print(f"  State: {issue['state']}")
        print(f"  Updated: {issue['updated_at']}")
        print(f"  URL: {issue['html_url']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <owner> <repo>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    
    issues = fetch_recent_issues(owner, repo)
    if issues:
        display_issues(issues)