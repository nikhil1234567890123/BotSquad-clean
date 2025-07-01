# main.py

from intent_classifier import classify_intent
from intent_map import intent_to_url
from search_links import search_links_within_intent_urls
from content_extractor import extract_pdf_text, extract_html_text
from generate_answer import generate_answer
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def main():
    query = input("User query: ")
    
    # Step 1: Classify Intent
    intent = classify_intent(query)
    print(f"[INFO] Detected Intent: {intent}")

    # Step 2: Get mapped URLs for that intent
    urls = intent_to_url.get(intent, "https://puchd.ac.in/")
    target_urls = urls if isinstance(urls, list) else [urls]

    # print(f"[INFO] Searching within {len(target_urls)} intent URLs...")
    
    # Step 3: Find links inside those pages that match the query
    sub_links = search_links_within_intent_urls(query, target_urls)

    if not sub_links:
        print("No specific links found matching your query.")
        return

    # print("\n🔗 Top Matching Links:")
    # for text, url in sub_links:
    #     print(f"- {text} → {url}")

    # Step 4: Extract content from the matched links
    context = ""

    for text, url in sub_links:
        print(f"\n[INFO] Extracting from: {url}")
        if url.endswith(".pdf"):
            extracted = extract_pdf_text(url)
        else:
            extracted = extract_html_text(url)

        if extracted:
            context += f"\n\n[{text.upper()}]\n{extracted[:2000]}..."  # Trim long content


    if context:
        answer = generate_answer(query, context)
        print("\nFINAL ANSWER:\n", answer)
    else:
        print("No meaningful content to generate answer.")


if __name__ == "__main__":
    main()
