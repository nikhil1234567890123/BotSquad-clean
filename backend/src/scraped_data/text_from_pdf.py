import requests
import fitz  # PyMuPDF
import csv
import os
from urllib.parse import urlparse
from time import sleep

# Load PDF links from txt file
with open("pdf_links.txt", "r", encoding="utf-8") as f:
    pdf_links = [line.strip() for line in f if line.strip()]

# Prepare CSV file
with open("pu_pdf_data.csv", "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filename", "url", "text"])  # header

    for i, url in enumerate(pdf_links, 1):
        try:
            print(f"📥 Processing {i}/{len(pdf_links)}: {url}")

            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://puchd.ac.in"
            }

            response = requests.get(url, headers=headers, timeout=15)

            # Skip if request failed or content is not a PDF
            if response.status_code != 200 or "application/pdf" not in response.headers.get("Content-Type", ""):
                print(f"❌ Skipping (status {response.status_code}): {url}")
                with open("failed_pdfs.txt", "a", encoding="utf-8") as fail_log:
                    fail_log.write(f"{url} - status {response.status_code}\n")
                continue

            # Create a filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            # Extract text from PDF
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Write extracted text to CSV
            csv_writer.writerow([filename, url, text.strip().replace("\n", " ")])

            sleep(0.5)  # be polite to server

        except Exception as e:
            print(f"❌ Exception for {url}: {e}")
            with open("failed_pdfs.txt", "a", encoding="utf-8") as fail_log:
                fail_log.write(f"{url} - exception: {e}\n")
            continue

print("✅ All PDFs processed and saved to pu_pdf_data.csv")

