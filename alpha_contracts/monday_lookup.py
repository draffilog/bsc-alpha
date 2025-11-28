import os
import sys
from typing import List, Optional
import httpx


MONDAY_API = "https://api.monday.com/v2"


def fetch_items(client: httpx.Client, token: str, board_id: int, page_size: int = 200, max_items: int = 3000):
    headers = {"Authorization": token, "Content-Type": "application/json"}
    q_cols = "query($bid:[ID!]!){ boards(ids:$bid){ columns{ id title type } } }"
    r_cols = client.post(MONDAY_API, headers=headers, json={"query": q_cols, "variables": {"bid": [board_id]}}, timeout=30)
    cols = (((r_cols.json() or {}).get("data") or {}).get("boards") or [{}])[0].get("columns") or []
    id_to_title = {c.get("id"): (c.get("title") or "") for c in cols}

    q_items = (
        "query($bid:[ID!]!,$limit:Int!,$cursor:String){ boards(ids:$bid){ items_page(limit:$limit,cursor:$cursor){ cursor items{ id name column_values{ id text } } } } }"
    )
    items = []
    cursor = None
    while len(items) < max_items:
        r = client.post(
            MONDAY_API,
            headers=headers,
            json={"query": q_items, "variables": {"bid": [board_id], "limit": page_size, "cursor": cursor}},
            timeout=60,
        )
        data = r.json()
        boards = (data.get("data", {}).get("boards") or [])
        if not boards:
            break
        page = boards[0].get("items_page") or {}
        items.extend(page.get("items") or [])
        cursor = page.get("cursor")
        if not cursor:
            break
    return id_to_title, items


def get_value(item: dict, id_to_title: dict, title: str) -> Optional[str]:
    target = title.strip().lower()
    # Find column id by title
    col_id = None
    for cid, t in id_to_title.items():
        if (t or "").strip().lower() == target:
            col_id = cid
            break
    if not col_id:
        return None
    for cv in item.get("column_values", []) or []:
        if cv.get("id") == col_id:
            return cv.get("text") or None
    return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="Find Monday item by domain and print Deal Owner")
    p.add_argument("--board-id", type=int, required=True)
    p.add_argument("--domain", type=str, required=True, help="Domain substring to match in Website column")
    p.add_argument("--name-like", type=str, default=None, help="Optional substring to match in Lead/Name")
    args = p.parse_args()

    token = os.getenv("MONDAY_TOKEN") or os.getenv("MONDAY_API_KEY")
    if not token:
        print("Missing MONDAY_TOKEN or MONDAY_API_KEY", file=sys.stderr)
        return 2

    with httpx.Client(follow_redirects=True) as client:
        id_to_title, items = fetch_items(client, token, args.board_id)

        domain = args.domain.lower().strip()
        name_like = (args.name_like or "").lower().strip()

        matches: List[dict] = []
        for it in items:
            website = get_value(it, id_to_title, "Website") or ""
            if domain and website.lower().find(domain) < 0:
                continue
            lead = get_value(it, id_to_title, "Lead") or (it.get("name") or "")
            if name_like and (lead.lower().find(name_like) < 0 and (it.get("name") or "").lower().find(name_like) < 0):
                continue
            matches.append({
                "id": it.get("id"),
                "name": it.get("name"),
                "lead": lead,
                "deal_owner": get_value(it, id_to_title, "Deal Owner") or get_value(it, id_to_title, "Owner") or "",
                "website": website,
            })

        for m in matches:
            print(f"id={m['id']} name={m['name']} lead={m['lead']} website={m['website']} deal_owner={m['deal_owner']}")

        if not matches:
            print("No matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



