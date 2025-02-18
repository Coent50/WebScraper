# Load packages and list of restaurants
import pandas as pd
from bs4 import BeautifulSoup
import requests

url = "https://www.yelp.com/biz/the-pink-door-seattle-4"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Yelp hides ratings inside meta tags
rating = soup.find("meta", {"itemprop": "ratingValue"})
if rating:
    print(f"Rating: {rating['content']} stars")
else:
    print("Could not find rating.")
