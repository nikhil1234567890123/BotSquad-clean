import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
import gc

def get_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)

def scrape_urls(urls, output_csv_file="pu_scraped_text.csv", max_size=2_000_000):
    with open(output_csv_file, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL", "Text"])  # Header

        for url in tqdm(urls, desc="Scraping Progress", unit="url"):
            if 'puchd.ac' in url:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                    }
                    response = requests.get(url, headers=headers, timeout=10)

                    # Skip large pages
                    if len(response.content) > max_size:
                        print(f"\nSkipping large page: {url}")
                        continue

                    response.raise_for_status()
                    text = get_visible_text(response.text)
                    writer.writerow([url, text])

                except Exception as e:
                    print(f"\nFailed to scrape {url}: {e}")

                # Force garbage collection
                gc.collect()

    print(f"\nScraping complete. Output saved to '{output_csv_file}'")

def run_scraper_with_chunking(input_txt_file, chunk_size=50):
    with open(input_txt_file, "r") as file:
        all_urls = [line.strip() for line in file if line.strip()]

    # Break URLs into chunks
    for i in range(0, len(all_urls), chunk_size):
        chunk_urls = all_urls[i:i + chunk_size]
        output_file = f"pu_scraped_part_{i//chunk_size + 1}.csv"
        print(f"\nProcessing chunk {i//chunk_size + 1} with {len(chunk_urls)} URLs")
        scrape_urls(chunk_urls, output_csv_file=output_file)

# Run it
run_scraper_with_chunking("internal_links.txt", chunk_size=500)
