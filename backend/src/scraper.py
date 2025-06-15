import requests
from bs4 import BeautifulSoup

url = "https://admissions.puchd.ac.in"  # ya jo bhi URL scrape kar rahe ho

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()

    with open("pu_admissions.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("Scraping successful! Data saved to pu_admissions.txt")
else:
    print(f" Failed to fetch data. Status code: {response.status_code}")
