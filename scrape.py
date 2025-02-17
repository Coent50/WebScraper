import selenium.webdriver as WD
from selenium.webdriver.chrome.service import Service
import time

def scrape_website (website):
    print("Launching chrome brwoser")

    chrome_driver_path ="./chromedriver"
    options = WD.ChromeOptions()
    driver = WD.Chrome(service=Service(chrome_driver_path), options=options)

    try:
        driver.get(website)
        print("Page loaded")
        html = driver.page_source
        time.sleep(10)

        return html
    finally:
        driver.quit()