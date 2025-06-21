import playwright
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # set to True to run headless
    page = browser.new_page()

        # 1. Go to the target URL
    page.goto("https://puchd.ac.in")

    links = page.query_selector_all("a")
    with open('pdf_links.txt', 'w', encoding='utf-8') as f:
        for link in links:
            href = link.get_attribute("href")
            if href and ".pdf" in href:
                f.write(href)
                f.write('\n')
