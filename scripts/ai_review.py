import argparse
import json
import os
import re
from typing import List, Tuple

import requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai may not be installed in test env
    OpenAI = None  # type: ignore

MAX_CHARS = 2000  # ~2K tokens approximation

SENSITIVE_PATTERNS = [re.compile(r"appsettings", re.IGNORECASE)]


def fetch_pr_diff(org: str, project: str, repo: str, pr_id: int, token: str) -> List[Tuple[str, str]]:
    """Fetch diffs for each changed file in a PR.

    Returns a list of tuples ``(path, diff)``.
    """
    headers = {"Authorization": f"Bearer {token}"}
    pr_url = f"{org}{project}/_apis/git/repositories/{repo}/pullRequests/{pr_id}?api-version=7.0"
    pr_resp = requests.get(pr_url, headers=headers)
    pr_resp.raise_for_status()
    pr = pr_resp.json()
    base = pr["lastMergeTargetCommit"]["commitId"]
    target = pr["lastMergeSourceCommit"]["commitId"]

    diff_url = (
        f"{org}{project}/_apis/git/repositories/{repo}/diffs/commits"
        f"?baseVersion={base}&targetVersion={target}&api-version=7.0&$top=200"
    )
    diff_resp = requests.get(diff_url, headers=headers)
    diff_resp.raise_for_status()
    diff_json = diff_resp.json()

    diffs = []
    for change in diff_json.get("changes", []):
        path = change.get("item", {}).get("path")
        if not path:
            continue
        if any(p.search(path) for p in SENSITIVE_PATTERNS):
            continue
        # Flatten diff hunks into unified diff text
        hunks = change.get("hunks") or []
        lines = []
        for h in hunks:
            header = f"@@ -{h['oldLine']},{h['oldLength']} +{h['newLine']},{h['newLength']} @@"
            lines.append(header)
            lines.extend(h.get("lines", []))
        diffs.append((path, "\n".join(lines)))
    return diffs


def chunk_diffs(diffs: List[Tuple[str, str]]) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for path, diff in diffs:
        entry = f"File: {path}\n{diff}\n"
        if size + len(entry) > MAX_CHARS and current:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(entry)
        size += len(entry)
    if current:
        chunks.append("\n".join(current))
    return chunks


def review_chunk(client: OpenAI, chunk: str) -> List[dict]:
    prompt = (
        "You are an automated code reviewer for .NET projects. "
        "Analyze the following diff and respond with JSON array containing "
        "objects with fields: filePath, line, comment.\n" + chunk
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data if isinstance(data, list) else data.get("comments", [])
    except Exception:
        return []


def post_comment(org: str, project: str, repo: str, pr_id: int, comment: dict, token: str) -> None:
    url = (
        f"{org}{project}/_apis/git/repositories/{repo}/pullRequests/{pr_id}/threads?api-version=7.0"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "comments": [{"content": comment["comment"], "commentType": 1}],
        "status": 1,
        "threadContext": {
            "filePath": comment["filePath"],
            "rightFileStart": {"line": comment.get("line", 1), "offset": 1},
        },
    }
    requests.post(url, headers=headers, json=body).raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-assisted code review")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--org", required=True)
    args = parser.parse_args()

    token = os.environ.get("SYSTEM_ACCESSTOKEN")
    if not token:
        raise RuntimeError("SYSTEM_ACCESSTOKEN is required")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if OpenAI is None:
        raise RuntimeError("openai package is required")
    client = OpenAI(api_key=api_key)

    diffs = fetch_pr_diff(args.org, args.project, args.repo, args.pr, token)
    if not diffs:
        print("No eligible files to review")
        return 0
    for chunk in chunk_diffs(diffs):
        comments = review_chunk(client, chunk)
        for c in comments:
            try:
                post_comment(args.org, args.project, args.repo, args.pr, c, token)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"Failed to post comment: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
