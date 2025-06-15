import requests
from bs4 import BeautifulSoup
import csv

# Step 1: Load PU links
with open(r"D:\BotSquad\backend\src\pu_links.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f.readlines()]

# Step 2: Create output CSV
with open("pu_pages_data.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["URL", "Title", "H1", "Contains Admission?"])  # You can add more fields

    # Step 3: Loop through links and extract content
    for index, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string.strip() if soup.title else "N/A"
                h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else "N/A"
                body_text = soup.get_text().lower()

                # Check for admission-related keywords
                keywords = ["admission", "apply", "prospectus", "course", "entrance", "eligibility"]
                admission_related = any(word in body_text for word in keywords)

                writer.writerow([url, title, h1, "Yes" if admission_related else "No"])
                print(f"{index+1}. Done: {url}")
            else:
                print(f"{index+1}. Failed: {url} [Status {response.status_code}]")
        except Exception as e:
            print(f"{index+1}. Error: {url} => {e}")
