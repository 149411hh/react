"""
Web page visiting and content extraction tools for the Research Agent.
Supports Jina Reader and fallback to httpx + BeautifulSoup + LLM summarization.
"""

import asyncio
import json
import os
from typing import List, Union

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from prompts import EXTRACTOR_PROMPT

# ====================== Configuration ======================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VISIT_TIMEOUT = 20
WEBCONTENT_MAXLENGTH = 50_000
SUMMARY_MODEL = "qwen-plus"

# Jina Reader Configuration
JINA_ENABLED = os.getenv("JINA_ENABLED", "1").strip() in ("1", "true", "yes")
JINA_API_KEYS = [k.strip() for k in os.getenv("JINA_API_KEYS", "").split(",") if k.strip()]
JINA_BASE = "https://r.jina.ai/"
JINA_TIMEOUT = 30

_jina_key_index = 0  # For key rotation

# LLM client for summarization
_summary_client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
)


# ====================== Fetch Functions ======================
async def fetch_page_jina(url: str) -> str:
    """Fetch webpage as clean markdown using Jina Reader API."""
    global _jina_key_index

    if not JINA_API_KEYS:
        return "[visit] No Jina API keys configured"

    tried = 0
    while tried < len(JINA_API_KEYS):
        key = JINA_API_KEYS[_jina_key_index % len(JINA_API_KEYS)]
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "text/markdown",
            "X-No-Cache": "true",
        }

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=JINA_TIMEOUT, verify=False) as client:
                resp = await client.get(f"{JINA_BASE}{url}", headers=headers)
                if resp.status_code == 200:
                    text = resp.text.strip()
                    return text if text else "[visit] Jina returned empty content"
                elif resp.status_code in (400, 403, 429):
                    _jina_key_index += 1
                    tried += 1
                    continue
                else:
                    return f"[visit] Jina HTTP {resp.status_code} for {url}"
        except Exception as e:
            return f"[visit] Jina fetch failed: {str(e)}"

    return "[visit] All Jina API keys exhausted"


async def fetch_page(url: str) -> str:
    """Fallback: Fetch webpage using httpx."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=VISIT_TIMEOUT, verify=False) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.text
            return f"[visit] HTTP {resp.status_code} error for {url}"
    except Exception as e:
        return f"[visit] Failed to fetch {url}: {str(e)}"


def extract_main_text(html: str) -> str:
    """Extract clean text from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    # Prefer <article> tag, fallback to body
    article = soup.find("article")
    text = article.get_text(separator="\n", strip=True) if article else soup.get_text(separator="\n", strip=True)

    # Clean excessive blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


# ====================== Summarization ======================
async def summarize_content(content: str, goal: str, max_retries: int = 2) -> str:
    """Use LLM to extract and summarize relevant information from webpage content."""
    content = content[:WEBCONTENT_MAXLENGTH]

    messages = [{
        "role": "user",
        "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
    }]

    for attempt in range(max_retries):
        try:
            resp = await _summary_client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()

            if not raw or len(raw) < 10:
                content = content[:int(len(content) * 0.7)]
                continue

            # Try to parse JSON output
            try:
                # Clean markdown code block if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                data = json.loads(raw)
                evidence = data.get("evidence", "")
                summary = data.get("summary", "")
                return f"Evidence in page:\n{evidence}\n\nSummary:\n{summary}"

            except json.JSONDecodeError:
                # Fallback: return raw content
                return f"Extracted content:\n{raw}"

        except Exception as e:
            print(f"[visit] Summarization attempt {attempt + 1} failed: {e}")

    return "The webpage content could not be processed."


# ====================== Visit Functions ======================
async def visit_page(url: str, goal: str) -> str:
    """Visit a single webpage and extract relevant information."""
    # Try Jina Reader first
    text = await fetch_page_jina(url) if JINA_ENABLED else "[visit] Jina disabled"

    # Fallback to httpx + BeautifulSoup
    if text.startswith("[visit]"):
        html = await fetch_page(url)
        if html.startswith("[visit]"):
            return f"The useful information in {url} for goal '{goal}' as follows:\n\n" \
                   f"Evidence: {html}\n\nSummary: Webpage could not be accessed."
        text = extract_main_text(html)

    if not text or len(text) < 20:
        return f"The useful information in {url} for goal '{goal}' as follows:\n\n" \
               f"Evidence: Empty or very short content.\n\nSummary: No extractable content."

    summary = await summarize_content(text, goal)
    return f"The useful information in {url} for goal '{goal}' as follows:\n\n{summary}"


async def visit_pages(urls: Union[str, List[str]], goal: str) -> str:
    """Visit multiple webpages concurrently."""
    if isinstance(urls, str):
        urls = [urls]

    async def _safe_visit(url: str) -> str:
        try:
            return await asyncio.wait_for(visit_page(url, goal), timeout=120)
        except asyncio.TimeoutError:
            return f"Error fetching {url}: visit timed out after 120s"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    results = await asyncio.gather(*[_safe_visit(url) for url in urls])
    return "\n=======\n".join(results)
