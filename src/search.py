"""
Self-contained search for Vernex using duckduckgo-search package.
No API keys, no Docker, no external services.
"""
import aiohttp
import asyncio
import html2text
import re

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

h2t = html2text.HTML2Text()
h2t.ignore_links = True
h2t.ignore_images = True
h2t.body_width = 0

def ddg_search(query, num_results=3):
    """Search DuckDuckGo."""
    if not DDGS:
        return []
    try:
        results = DDGS().text(query, max_results=num_results)
        return [r.get("href") for r in results if r.get("href")]
    except Exception as e:
        print(f"DDG error: {e}")
        return []

async def fetch_page(session, url):
    """Fetch and convert page to text using Jina Reader (for clean MD) with local fallback."""
    jina_url = f"https://r.jina.ai/{url}"
    
    # Strategy 1: Jina Reader (Best for LLM comprehension)
    try:
        async with session.get(jina_url, timeout=10, ssl=False) as resp:
            if resp.status == 200:
                text = await resp.text()
                if len(text) > 100: # Ensure we got actual content
                    return {"url": url, "content": text[:2000], "ok": True, "method": "jina"}
    except:
        pass
    
    # Strategy 2: Direct Fetch + html2text (Fallback)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        async with session.get(url, timeout=8, headers=headers, ssl=False) as resp:
            if resp.status == 200:
                html = await resp.text()
                text = h2t.handle(html)
                text = re.sub(r'\n{3,}', '\n\n', text)
                return {"url": url, "content": text[:1500], "ok": True, "method": "local"}
    except Exception as e:
        print(f"Fallback fetch failed for {url}: {e}")
        
    return {"url": url, "content": "", "ok": False, "method": "failed"}

async def fetch_all(urls):
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        return await asyncio.gather(*[fetch_page(session, u) for u in urls])

def search_and_fetch(query, num_results=3):
    """
    Main search function. Fully self-contained.
    """
    print(f"[SEARCH] Querying: {query}")
    
    urls = ddg_search(query, num_results)
    
    if not urls:
        return "No search results found."
    
    print(f"[SEARCH] Fetching {len(urls)} pages...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pages = loop.run_until_complete(fetch_all(urls))
    except Exception as e:
        return f"Fetch failed: {e}"
    
    output = []
    for p in pages:
        if p["ok"] and p["content"]:
            output.append(f"SOURCE: {p['url']}\n{p['content'][:1000]}\n---")
    
    return "\n".join(output) if output else "Search found URLs but couldn't extract content."

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "JUCE audio framework"
    print(search_and_fetch(q))
