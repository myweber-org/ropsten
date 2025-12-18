import requests
import sys
from datetime import datetime, timedelta

def fetch_recent_issues(owner, repo, days=7):
    """
    Fetch recent issues from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    params = {
        'state': 'all',
        'since': since_date,
        'per_page': 10,
        'sort': 'created',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        return issues
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return []

def display_issues(issues):
    """
    Display a list of issues in a formatted way.
    """
    if not issues:
        print("No recent issues found.")
        return

    print(f"Found {len(issues)} issue(s):\n")
    for issue in issues:
        issue_number = issue.get('number', 'N/A')
        title = issue.get('title', 'No Title')
        state = issue.get('state', 'unknown')
        user_login = issue.get('user', {}).get('login', 'Unknown')
        created_at = issue.get('created_at', '')
        if created_at:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created_str = created_dt.strftime('%Y-%m-%d %H:%M')
        else:
            created_str = 'N/A'

        print(f"#{issue_number} [{state.upper()}] {title}")
        print(f"   Opened by {user_login} on {created_str}")
        print(f"   URL: {issue.get('html_url', 'N/A')}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_github_issues.py <owner> <repo> [days]")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    print(f"Fetching recent issues from {owner}/{repo} (last {days} days)...")
    issues = fetch_recent_issues(owner, repo, days)
    display_issues(issues)