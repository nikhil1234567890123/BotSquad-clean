from playwright.sync_api import sync_playwright

def get_links(page):
    pdfs = []
    normal = []

    links = page.query_selector_all("a")
    for link in links:
        href = link.get_attribute("href")
        if not href:
            continue

        if href.startswith("/"):
            href = "https://puchd.ac.in" + href

        if href.endswith(".pdf"):
            pdfs.append(href)
        else:
            normal.append(href)

    with open('pdf_links.txt', 'a', encoding='utf-8') as f:
        for i in pdfs:
            f.write(i + '\n')

    with open('internal_links.txt', 'a', encoding='utf-8') as f:
        for i in normal:
            f.write(i + '\n')

    print(f"Collected {len(pdfs)} PDF links and {len(normal)} normal links.")
    
    return normal


def open_links():
    pass


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    page.goto("https://puchd.ac.in", wait_until="domcontentloaded", timeout=60000)

    visited_links = []
    list_of_links = get_links(page)

    for l in list_of_links:
        if ('.ac' in l) and l not in visited_links:
            try:
                print(f"Visiting: {l}")
                page.goto(l, timeout=10000)
                get_links(page)
                visited_links.append(l)
            except Exception as e:
                print(f"Timeout or error at {l}: {e}")
                continue

    print(len(visited_links))