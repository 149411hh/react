"""
Search tools module for the Research Agent.
Provides web search capabilities using Serper (Google) and IQS (Bing-like) engines.
"""

import json
import os
import re
import urllib.parse
from typing import List, Optional

import httpx

# ====================== Configuration ======================
IQS_API_KEY = os.getenv("IQS_API_KEY", "")
IQS_BASE = "https://cloud-iqs.aliyuncs.com"
IQS_TIMEOUT = 15

SERPER_API_KEYS = [
    k.strip() for k in os.getenv("SERPER_API_KEYS", "").split(",") if k.strip()
]
SERPER_BASE = "https://google.serper.dev/search"
SERPER_TIMEOUT = 15

_serper_key_index = 0  # For key rotation


# ====================== Helper Functions ======================
def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


def _simplify_query(query: str) -> str:
    """Simplify complex query by removing modifiers."""
    if not query or len(query) < 10:
        return query

    simplified = re.sub(r'的|which|that|whose|who|when|where', ' ', query, flags=re.IGNORECASE)
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    # Avoid over-simplification
    if len(simplified) < len(query) * 0.3:
        return query

    return simplified if simplified != query else query


def _is_poor_result(result: str) -> bool:
    """Determine if search result is poor quality."""
    if not result or len(result) < 100:
        return True
    if any(keyword in result for keyword in ["No results found", "Search failed", "found 0 results"]):
        return True
    return False


# ====================== Search Implementations ======================
def iqs_search(query: str, retries: int = 3) -> str:
    """Perform search using IQS (Bing-like) engine."""
    headers = {"X-API-Key": IQS_API_KEY}
    params = {"query": query, "timeRange": "NoLimit"}

    for attempt in range(retries):
        try:
            resp = httpx.get(
                f"{IQS_BASE}/search/genericSearch",
                headers=headers,
                params=params,
                timeout=IQS_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                return _format_iqs_results(query, data.get("pageItems", []))
            else:
                print(f"[search] IQS error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[search] IQS attempt {attempt + 1} failed: {e}")

    return f"Search failed for '{query}'. Please try a different query."


def serper_search(query: str) -> str:
    """Perform search using Serper (Google) engine with key rotation."""
    global _serper_key_index

    if not SERPER_API_KEYS:
        print("[search] No Serper keys configured, falling back to IQS")
        return iqs_search(query)

    tried = 0
    while tried < len(SERPER_API_KEYS):
        key = SERPER_API_KEYS[_serper_key_index % len(SERPER_API_KEYS)]
        headers = {"X-API-Key": key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 10}

        try:
            resp = httpx.post(SERPER_BASE, headers=headers, json=payload, timeout=SERPER_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return _format_serper_results(query, data)

            elif resp.status_code in (400, 403, 429):
                print(f"[search] Serper key exhausted (HTTP {resp.status_code}), rotating")
                _serper_key_index += 1
                tried += 1
                continue
            else:
                print(f"[search] Serper error {resp.status_code}")
                break
        except Exception as e:
            print(f"[search] Serper request failed: {e}")
            break

    print("[search] All Serper keys failed, falling back to IQS")
    return iqs_search(query)


def _format_serper_results(query: str, data: dict) -> str:
    """Format Serper results into consistent style."""
    organic = data.get("organic", [])
    if not organic:
        return f"No results found for '{query}'."

    snippets = []
    for idx, item in enumerate(organic, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        date = item.get("date", "")

        entry = f"{idx}. [{title}]({link})"
        if date:
            entry += f"\nDate: {date}"
        if snippet:
            entry += f"\n{snippet}"

        snippets.append(entry)

    return f"Search for '{query}' found {len(snippets)} results:\n\n" + "\n\n".join(snippets)


def _format_iqs_results(query: str, page_items: list) -> str:
    """Format IQS results into consistent style."""
    if not page_items:
        return f"No results found for '{query}'."

    snippets = []
    for idx, item in enumerate(page_items, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "") or item.get("htmlSnippet", "")

        entry = f"{idx}. [{title}]({link})"
        if snippet:
            entry += f"\n{snippet}"

        snippets.append(entry)

    return f"Search for '{query}' found {len(snippets)} results:\n\n" + "\n\n".join(snippets)


# ====================== Main Search Function ======================
def batch_search(queries: List[str], engines: Optional[List[str]] = None) -> str:
    """
    Perform batch search with automatic engine selection and fallback logic.
    
    Args:
        queries: List of search queries
        engines: Optional list of engines ('google' or 'bing') for each query
    
    Returns:
        Combined search results as string
    """
    if isinstance(queries, str):
        queries = [queries]
    if engines is None:
        engines = [None] * len(queries)

    results = []
    for q, engine in zip(queries, engines):
        # Determine engine: LLM choice > language auto-detect
        if engine == "google":
            use_google = True
        elif engine == "bing":
            use_google = False
        else:
            use_google = not _contains_chinese(q)

        engine_name = "Google/Serper" if use_google else "Bing/IQS"
        print(f'[search] Query: "{q[:60]}..." → {engine_name}')

        # Primary search
        result = serper_search(q) if use_google else iqs_search(q)

        # Fallback if result is poor
        if _is_poor_result(result):
            fallback_engine = "Bing/IQS" if use_google else "Google/Serper"
            print(f"[search] Poor results, trying {fallback_engine} fallback...")
            fallback = iqs_search(q) if use_google else serper_search(q)

            if not _is_poor_result(fallback):
                result = fallback
            else:
                # Try simplified query as last resort
                simplified = _simplify_query(q)
                if simplified != q:
                    print(f"[search] Trying simplified query: {simplified}")
                    result = serper_search(simplified) if use_google else iqs_search(simplified)

        results.append(result)

    return "\n=======\n".join(results)
