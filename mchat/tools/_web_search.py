import logging
import time
from io import BytesIO
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from config import settings


def google_search(
    query: str, num_results: int = 5, max_chars: int = 1000, log_file: str | None = None
) -> List[Dict]:
    """
    Perform a Google Custom Search and fetch enriched results.

    Args:
        query (str): Search query.
        num_results (int): Number of results to fetch (default: 5).
        max_chars (int): Maximum characters to extract from the content of each URL.
        log_file (str): File to log the results (default: "google_search.log").

    Returns:
        List[Dict]: A list of enriched search results.
    """
    # Fetch API key and Search Engine ID
    api_key = settings.get("google_api_key", None)
    search_engine_id = settings.get("google_search_engine_id", None)
    log_file = (
        settings.get("google_search_log_file", None) if log_file is None else log_file
    )

    if not api_key or not search_engine_id:
        raise ValueError(
            "API key or Search Engine ID not found in environment variables."
        )

    # Google Custom Search API endpoint
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": query, "num": num_results}

    try:
        # Perform the API request
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        results = response.json().get("items", [])

        # Helper function to extract content from a URL
        def get_page_content(page_url: str) -> str:
            try:
                # Check if the URL points to a PDF
                head_response = requests.head(
                    page_url, timeout=10, allow_redirects=True
                )
                content_type = head_response.headers.get("Content-Type", "")

                if "application/pdf" in content_type:
                    # Fetch the PDF content
                    pdf_response = requests.get(page_url, timeout=10)
                    pdf_response.raise_for_status()
                    pdf_content = BytesIO(pdf_response.content)

                    # Extract text from the PDF
                    reader = PdfReader(pdf_content)
                    pdf_text = ""
                    for i, page in enumerate(reader.pages):
                        if i >= 2:  # Limit to the first 2 pages for performance
                            break
                        pdf_text += page.extract_text()

                    pdf_text = pdf_text[:max_chars].strip()

                    # Log PDF content
                    if log_file:
                        with open(log_file, "a") as f:
                            f.write(f"PDF URL: {page_url}\n")
                            f.write(f"PDF Content: {pdf_text}\n\n")

                    return pdf_text

                # If not a PDF, fetch the webpage content
                page_response = requests.get(page_url, timeout=10)
                page_response.raise_for_status()
                soup = BeautifulSoup(page_response.content, "html.parser")

                # Remove unwanted elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()

                # Extract text with a character limit
                text = soup.get_text(separator=" ", strip=True)
                out = text[:max_chars].strip()

                # Log webpage content
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"URL: {page_url}\n")
                        f.write(f"Content: {out}\n\n")

                return out

            except requests.RequestException as req_error:
                logging.warning(
                    f"Error fetching page content from {page_url}: {req_error}"
                )
                return "Error fetching content."
            except Exception as ex:
                logging.warning(
                    f"Unexpected error fetching content from {page_url}: {ex}"
                )
                return "Error fetching content."

        # Enrich results with page content
        enriched_results = []
        for item in results:
            body = get_page_content(item["link"])
            enriched_results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "body": body,
                }
            )
            time.sleep(1)  # Be respectful to the servers (rate limiting)

        return enriched_results

    except requests.RequestException as e:
        logging.error(f"API request error: {e}")
        raise RuntimeError(f"Error in Google Custom Search API request: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise RuntimeError(f"An unexpected error occurred: {e}") from e
