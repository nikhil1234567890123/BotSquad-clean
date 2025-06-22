import requests
from bs4 import BeautifulSoup
import csv

def get_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and noscript tags
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    # Extract and clean visible text
    visible_text = soup.get_text(separator="\n", strip=True)
    return visible_text

def scrape_and_save_to_csv(input_txt_file, output_csv_file="pu_scraped_text.csv"):
    with open(input_txt_file, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    with open(output_csv_file, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL", "Text"])  # CSV header

        for url in urls:
            if 'puchd.ac' in url:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    print(f"Scraping: {url}")
                    text = get_visible_text(response.text)
                    writer.writerow([url, text])
                except Exception as e:
                    print(f"Failed to scrape {url}: {e}")


    print(f"\nScraping complete. Data saved to '{output_csv_file}'")

# Example usage
scrape_and_save_to_csv("internal_links.txt")
