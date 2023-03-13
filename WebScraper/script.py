import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

# uncomment for logs
# service_log_path = "chromedriver.log"

# Start a web driver and navigate to Google
# driver = webdriver.Chrome(options=options, service_log_path=service_log_path)
driver = webdriver.Chrome(options=options)
driver.get("https://www.google.com")

# Input the keyword to search
search_input = driver.find_element(By.NAME, "q")
keyword = input("Enter a keyword to search: ")
keyword = keyword + "stores"
search_input.send_keys(keyword)
search_input.send_keys(Keys.RETURN)

# Get different pages in google search
pages_a_tags = driver.find_elements(By.CSS_SELECTOR, "a.fl")

page_urls = []
for a in pages_a_tags:
    page_urls.append(a.get_attribute("href"))

print("google_page_urls")

for index, page_url in enumerate(page_urls):
    print("page", index + 2, page_url)
    
# Get anchor tags in page 1
a_tags = driver.find_elements(By.CSS_SELECTOR, "a")

print("page 1 urls")

# Get list of urls to visit in page 1
urls = set()
for a in a_tags:
    href = a.get_attribute("href")
    if href and ("google" not in href) and ("amazon" not in href):
        if href not in urls:
            print("Adding", href)
            urls.add(href)

# Get list of urls to visit in other google pages
for page_no, page_url in enumerate(page_urls):
    print("page", page_no + 2, "urls")
    driver.get(page_url)
    
    # Get anchor tags in page 
    a_tags = driver.find_elements(By.CSS_SELECTOR, "a")

    # Get list of urls to visit in page
    for a in a_tags:
        href = a.get_attribute("href")
        if href and ("google" not in href) and ("amazon" not in href):
            if href not in urls:
                print("Adding", href)
                urls.add(href)
    
print("Number of urls:", len(urls))
    
# Create a folder for the screenshots if it doesn't exist
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)
    
def save_screenshot(driver: webdriver.Chrome, path: str = '/tmp/screenshot.png') -> None:
    # Ref: https://stackoverflow.com/a/52572919/
    original_size = driver.get_window_size()
    required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
    driver.set_window_size(required_width, required_height)
    # driver.save_screenshot(path)  # has scrollbar
    driver.find_element(By.TAG_NAME, 'body').screenshot(path)  # avoids scrollbar
    driver.set_window_size(original_size['width'], original_size['height'])

for url in urls:
    print(url)
    
# Get 50 urls as a list
urls50 = sorted(list(urls))[:50]

# Visit each website and take a screenshot
for i, url in enumerate(urls50):
    driver.get(url)
    time.sleep(5)
    height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(1920, height)
    time.sleep(2)
    # save_screenshot(driver, f"{screenshot_folder}/screenshot{i}.png")
    driver.save_screenshot(f"{screenshot_folder}/screenshot{i}.png")

# Close the web driver
driver.close()