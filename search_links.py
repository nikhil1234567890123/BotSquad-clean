# search_links.py

from playwright.sync_api import sync_playwright

def search_links_within_intent_urls(query, urls):
    query_keywords = query.lower().split()
    matched_links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for url in urls:
            try:
                page.goto(url, timeout=20000)
                anchors = page.query_selector_all("a")

                for a in anchors:
                    href = a.get_attribute("href")
                    text = a.inner_text().strip().lower()

                    if not href or not text:
                        continue
                    if any(kw in text for kw in query_keywords):
                        full_url = href if href.startswith("http") else url.rsplit("/", 1)[0] + "/" + href.lstrip("/")
                        matched_links.append((text, full_url))
            
            except Exception as e:
                print(f"[WARN] Could not open {url}: {e}")

        browser.close()
    
    # Remove duplicates
    seen = set()
    unique_links = []
    for text, url in matched_links:
        if url not in seen:
            seen.add(url)
            unique_links.append((text, url))

    return unique_links[:5]  # top 5 relevant
