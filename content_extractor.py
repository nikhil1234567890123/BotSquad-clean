# content_extractor.py

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import io

def extract_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Load PDF directly from memory
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text.strip()

    except Exception as e:
        print(f"[ERROR] Failed to extract PDF from memory: {e}")
        return ""

def extract_html_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        content = []

        for tag in soup.find_all(["p", "li", "td", "h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 30:
                content.append(text)

        return "\n".join(content)

    except Exception as e:
        print(f"[ERROR] Failed to extract HTML: {e}")
        return ""
