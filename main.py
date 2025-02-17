import streamlit as st 
from scrape import scrape_website

st.title ("Web scraper")
url = st.text_input("Enter a website URL")

if st.button ("Scrape website"):
    st.write("Scraping in progress")
    result = scrape_website(url)
    print(result)
    


