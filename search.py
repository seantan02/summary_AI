from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def get_stackoverflow_url(search_key, headless = True) -> str:
    """
    This function searches for the first stackoverflow link given a search key
    """

    options = webdriver.ChromeOptions()
    options.add_argument('window-size=800x841')

    if headless:
        options.add_argument('headless')
    
    driver = webdriver.Chrome(options=options)

    # Open Google
    driver.get('https://www.google.com')

    time.sleep(5)

    # Find the search box using its name attribute value
    search_box = driver.find_element(By.NAME, 'q')

    # Input the search query
    search_box.send_keys(search_key)

    # Press ENTER to perform the search
    search_box.send_keys(Keys.RETURN)

    time.sleep(2)

    first_overflow_link = ""

    # contains the search results
    searchResults = driver.find_elements(By.CLASS_NAME, 'g')

    time.sleep(2)
    
    for result in searchResults:
        element = result.find_element(By.CSS_SELECTOR, 'a')

        link = element.get_attribute('href')

        header = result.find_element(By.CSS_SELECTOR, 'h3').text
        # if 'stackoverflow.com' in link:
        #     all_stack_overflow.append({
        #         'header' : header, 'link' : link
        #     })
        if 'stackoverflow.com' in link:
            first_overflow_link = link
            break

    time.sleep(2)

    # Close the browser
    driver.quit()

    return first_overflow_link

def get_stackoverflow_best_answer(url, headless = True):
    """
    This function finds the best answer given a stackoverflow link by:
    1. Get the accepted answer (that has a tick), if any
    2. Get the most voted answer if no accepted answer found (That includes first answer if no accepted answer)
    """

    options = webdriver.ChromeOptions()
    options.add_argument('window-size=800x841')

    if headless:
        options.add_argument('headless')
    
    driver = webdriver.Chrome(options=options)

    # Open Google
    driver.get(url)

    time.sleep(5)

    try:  # Get the accepted answer
        best_answer = driver.find_element(By.CLASS_NAME, 'accepted-answer')
    except Exception:  # No accepted answer
        highest_vote = None
        best_answer = None

        all_answers = driver.find_elements(By.CLASS_NAME, 'answer')

        for answer in all_answers:
            vote = int(answer.get_attribute("data-score").strip())

            if highest_vote is None or vote > highest_vote:
                highest_vote = vote
                best_answer = answer

    answer_cell = best_answer.find_element(By.CLASS_NAME, 'answercell')
    answer_content = answer_cell.find_elements(By.TAG_NAME, 'div')[0].text
    
    time.sleep(2)

    return answer_content