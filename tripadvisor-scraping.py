from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import os
import sys
import time
import csv
import argparse
import pandas as pd    
import hashlib
import base64
import re
import random
import string
from collections import OrderedDict
import json
import pickle


# define a matchers dict and a processors dict to get the user info in the user overlay through RegEx; note that the parentheses define the part taken
matchers = {    
    "user_ta_level": lambda item: re.search("Level ([0-9]+)", item),
    "user_age_range": lambda item: re.search("[0-9][0-9]-[0-9][0-9]", item) or re.search("[0-9][0-9]\+", item),
    "user_sex": lambda item: re.search("woman", item) or re.search("man", item),
    "user_location": lambda item: re.search("[fF]rom ([a-zA-Z]+, [a-zA-Z]+)", item),
    "user_nb_contributions": lambda item: re.search("([0-9]+) Contributions", item),
    "user_nb_cities_visited": lambda item: re.search("([0-9]+) Cities visited", item),
    "user_nb_helpful_votes": lambda item: re.search("([0-9]+) Helpful votes", item),
    "user_nb_photos": lambda item: re.search("([0-9]+) Photos", item)
}

def get_match(index=0):
    return lambda match: match.group(index)

processors = {
    "user_ta_level": get_match(1),
    "user_age_range": get_match(),
    "user_sex": get_match(),
    "user_location": get_match(1),
    "user_nb_contributions": get_match(1),
    "user_nb_cities_visited": get_match(1),
    "user_nb_helpful_votes": get_match(1),
    "user_nb_photos": get_match(1),
}

def parse_user_info(items):
    output = OrderedDict({k: 'N/A' for k in basic_user_info_header})
    output['user_name'] = items[0]  # the first item is always the user name
    # output['user_name_hash'] = base64.b64encode(hashlib.md5(bytes(output['user_name'], 'utf-8')).digest())    # hash the user name to anonymize it if needed

    for item in items:
        for key, matcher in matchers.items():
            if output[key] != 'N/A':    # initialize the user data row that will be written in the csv file
                continue
            
            m = matcher(item)   # match the item with the RegEx
            if m:   # if the item matched the RegEx
                output[key] = processors[key](m)
            
    return output


def start_scraping(is_resume_scraping, scraping_state, data_writer):

    # try:

    print("Resume scraping: " + str(is_resume_scraping))
    print("The initial scraping state is: ")
    print(scraping_state)

    # find in city_names the index of the city from which to resume scraping if it got interrupted; if it did not, scrape from the normal initial city
    resume_city_index = scraping_state['cities_to_scrape'].index(scraping_state['city_resume'])
    cities_to_scrape = scraping_state['cities_to_scrape'][resume_city_index:scraping_state['nb_of_cities_to_scrape']]
    
    print("Cities to scrape: ")
    print(cities_to_scrape)

    # iterate over the cities to scrape
    for city_index, review_city_name in enumerate(cities_to_scrape):

        if not is_resume_scraping:
            print("\nWe start scraping the restaurant reviews in: " + review_city_name)

            # save in the scraping state the current city name and update the list of cities to scrape
            scraping_state['city_resume'] = review_city_name
            scraping_state['cities_to_scrape'] = cities_to_scrape[city_index:]
            
            # open the website
            driver.get(scraping_state['url_main'])  
            # scale_element = driver.find_element(by=By.TAG_NAME, value="body")
            # driver.execute_script("arguments[0].style.transform='scale(1)';", scale_element)

            time.sleep(4)

            # click on the "Restaurants" button
            restaurants_button = driver.find_element(by=By.XPATH, value="//a[@href='/Restaurants']")
            restaurants_button.click()

            time.sleep(3)

            # click on the "Search" bar and input the city name
            search_city_bar = driver.find_element(by=By.XPATH, value="//div[@class='slvrn Z0 Wh rsqqi EcFTp GADiy']//input[@placeholder='Where to?']")
            search_city_bar.click()
            search_city_bar.send_keys(scraping_state['city_resume'])

            time.sleep(3)

            # click on the selected city (which is the first suggestion)
            # search_city_bar.send_keys(Keys.ENTER) # doesn't work because it does not redirect to the top restaurants of the city if we just press enter. We have to click on the suggested option!
            selected_city = driver.find_element(by=By.XPATH, value="//div[@class='XYHql z RJdtB']")
            selected_city.click()

            time.sleep(4)

            # scroll down to reach the restaurants top reviews
            top_restaurants_title = driver.find_element(by=By.XPATH, value="//div[@class='pFMac b Cj']")
            driver.execute_script(
            "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
            top_restaurants_title)
            
            time.sleep(2)

            # since it is a new city, get the few first top restaurants in the city to later iterate over them
            top_restaurants_in_city_container = driver.find_elements(by=By.XPATH, value=".//div[@class='RfBGI']/span/a")
            scraping_state['restaurants_urls_resume'] = [top_restaurant_in_city.get_attribute('href') for top_restaurant_in_city in top_restaurants_in_city_container]
            print(f"{len(top_restaurants_in_city_container)} top restaurants found on this page for the city {scraping_state['city_resume']}")
            # print(top_restaurants_in_city_container[0].text)

            # save in a csv file the restaurants urls for a city
            with open(f"./Data/TA_{scraping_state['city_resume'].split(',')[0].replace(' ', '-')}_restaurants_urls.csv", "w") as file:
                    csv_writer_restaurant = csv.writer(file, delimiter="\t", lineterminator="\n")
                    csv_writer_restaurant.writerow(["city", "restaurant_url"])
                    for restaurant_url in scraping_state['restaurants_urls_resume']:
                        csv_writer_restaurant.writerow([scraping_state['city_resume'], restaurant_url])

            # get the restaurants urls to scrape (potentially a subset of the top restaurants in the city, unless -1 value)
            scraping_state['restaurants_urls_resume'] = scraping_state['restaurants_urls_resume'][:scraping_state['nb_of_restaurants_to_scrape_per_city']]


        # get the restaurants for which to continue scraping if the scraping got interrupted; if it did not, the restaurants would have been gotten from the page with the top restaurants in the city
        restaurants_urls_resume = scraping_state['restaurants_urls_resume']

        # iterate over the restaurants (urls) to scrape
        for restaurant_index, restaurant_url in enumerate(restaurants_urls_resume):  # only select the first 3 top restaurants for now
            
            print("Currently scraping restaurant: ", restaurant_url)

            # save in the scraping state the urls of the restaurants that remain to be scraped in case the scraping is interrupted
            scraping_state['restaurants_urls_resume'] = scraping_state['restaurants_urls_resume'][restaurant_index:]

            if is_resume_scraping:
                driver.get(scraping_state['webpage_resume'])
            else:
                driver.get(restaurant_url)
        
            time.sleep(4)
            
            # scroll down to reach the general reviews panel
            dummy_reviews_point_for_scroll = driver.find_element(by=By.XPATH, value="//div[@class='title_text']")
            driver.execute_script(
            "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
            dummy_reviews_point_for_scroll)

            time.sleep(3)

            try: 
                # remove the sticky bar to avoid clicking on it
                driver.execute_script("""
                var l = document.getElementsByClassName("stickyBar")[0];
                l.parentNode.removeChild(l);
                """)
            except NoSuchElementException:
                print("Problem removing the sticky bar")

            # Select the language of the reviews. TODO: See if it can only be done once per city; but I think not without cookies.
            language_id_selector = "filters_detail_language_filterLang_" + language_to_scrape

            try:
                # click on the wanted language in the list of languages directly available
                time.sleep(3)
                selected_lang = driver.find_element(by=By.XPATH, value=f'.//div[@class="item"][@data-value={language_to_scrape}]/label[@class="label container"][@for={language_id_selector}]')
                # selected_lang = driver.find_element(by=By.XPATH, value=f'.//div[@class="item"][@data-value={language_to_scrape}]')
                # selected_lang = driver.find_element(by=By.XPATH, value=f'.//label[@class="label container"][@for={language_id_selector}]')
                # time.sleep(8)
                # driver.implicitly_wait(5)
                # ActionChains(driver).move_to_element(selected_lang).click(selected_lang).perform()
                # selected_lang.click() 
                # WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, language_id_selector))).click()
                # selected_lang = driver.find_element(by=By.XPATH, value=f'/html[1]/body[1]/div[2]/div[2]/div[2]/div[7]/div[1]/div[1]/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[4]/div[1]/div[2]/div[1]/div[3]/label[1]/input[1]')
                selected_lang.click() 
                language_is_selected = True     # from now the language is selected and does not have to be selected again until the next city
                time.sleep(3)
            except NoSuchElementException:
                try:
                    # click on the "More languages" button to get the overlay with more language choices
                    more_languages_bar = driver.find_element(by=By.XPATH, value='.//div[@class="taLnk"]')
                    more_languages_bar.click()
                    time.sleep(3)
                    # click on the wanted language in the "More languages" overlay
                    language_in_bar = driver.find_element(by=By.ID, value=language_id_selector)
                    time.sleep(1)
                    driver.execute_script(
                        "arguments[0].click();",
                        language_in_bar)
                    # language_in_bar.click()
                    language_is_selected = True     # from now the language is selected and does not have to be selected again until the next city
                    time.sleep(3)
                except NoSuchElementException:  # go to the next restaurant              
                    continue

            # make sure to scroll down before getting the reviews
            try:
                dummy_showreviews_point_for_scroll = driver.find_element(by=By.XPATH, value="//div[normalize-space()='Show reviews that mention']")
                driver.execute_script(
                "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                dummy_showreviews_point_for_scroll)
            except NoSuchElementException:
                try:
                    dummy_firstreview_point_for_scroll = driver.find_element(by=By.XPATH, value=".//div[@class='review-container']")
                    driver.execute_script(
                    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                    dummy_firstreview_point_for_scroll)
                except NoSuchElementException:
                    print("No reviews found for this restaurant")
                    continue    # go to the next restaurant
            
            time.sleep(3)

            scraping_state["total_restaurant_users_added"] = 0

            time.sleep(3)


            # iterate over the restaurant's pages
            for page_index in range(scraping_state['page_nb_resume']-1, scraping_state["nb_of_pages_to_scrape_per_restaurant"]):

                # get the current url and save it as this is the url from which we should start again if the scraping run is interrupted; get the current page index
                scraping_state['webpage_resume'] = driver.current_url
                scraping_state['page_nb_resume'] = page_index
                
                time.sleep(3)

                # expand the review
                reviews_container = driver.find_elements(by=By.XPATH, value=".//div[@class='review-container']")

                print(f"{len(reviews_container)} reviews found on page {page_index+1}")
                
                for review_index, element in enumerate(reviews_container):   # a container element is a user review
                    print(f"Entered in review {review_index+1} from page {page_index+1}")

                    # If the user review is a review translated from English to another language (e.g., zhCH), then skip it (TODO: see if still need to collect it or not)
                    # if not element.find_elements(by=By.XPATH, value="//div[@data-prwidget-name='reviews_google_translate_button_hsx']"):    # using xpath to find the element. find_elements returns empty list (which is equal to False)
                    #     continue

                    # Open all the reviews "More..." on the page to get the full review
                    review_expander_buttons = element.find_elements(by=By.XPATH, value="//span[@class='taLnk ulBlueLinks']")
                    time.sleep(1)
                    if review_expander_buttons:
                        review_expander_buttons[0].click()
                        time.sleep(2)  

                    # a user may have a deactivated account, in which case the user info overlay pop up will not appear
                    try:
                        overlay_element = element.find_element(by=By.XPATH, value=".//div[@class='memberOverlayLink clickable']")    # user overlay to make the user info pop up appear on the same page
                        time.sleep(2)
                    except NoSuchElementException:
                        print("No user info found for this review. Account has been deleted.")
                        continue

                    # ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
                    # overlay_element = WebDriverWait(driver, 5,ignored_exceptions=ignored_exceptions).until(expected_conditions.presence_of_element_located((By.XPATH, ".//div[@class='memberOverlayLink clickable']")))
                    overlay_element.click()
                    time.sleep(2)

                    user_overlay_info = driver.find_element(by=By.XPATH, value=".//div[@class='memberOverlay simple container moRedesign']") 
                    user_info_element = user_overlay_info.find_element(by=By.XPATH, value=".//div[@class='memberOverlayRedesign g10n']")
                    user_info = user_info_element.text.split("\n")        # get the user info   # E.g: ['UserName', 'Level 4 Contributor', 'Send Message', 'Tripadvisor member since 2007', '35-49 man from Wednesbury, United Kingdom', ' 37 Contributions', ' 38 Cities visited', ' 28 Helpful votes', ' 14 Photos', 'Like a Local Thrill Seeker Shopping Fanatic Peace and Quiet Seeker', 'View all', 'REVIEW DISTRIBUTION', 'Excellent\t', '\t32', 'Very good\t', '\t2', 'Average\t', '\t2', 'Poor\t', '\t0', 'Terrible\t', '\t1']
                    
                    # Parse the user info according to the hardcoded keywords in the list
                    output = parse_user_info(user_info)

                    # Get the user tags
                    user_tags = user_overlay_info.find_elements(by=By.XPATH, value=".//a[@class='memberTagReviewEnhancements']")
                    nb_user_tags = len(user_tags)
                    user_tags = ";".join([tag.text for tag in user_tags]) if nb_user_tags else "N/A"  # we arbritrarily define semicolon as the separator within the feature user_tags
                    # print("User tags: ", user_tags)
                    
                    # If the UP is complete enough; TODO: see what can be a good enough condition (e.g., if we should also use the number of tags as part of the condition although it may be too restrictive)
                    user_info_condition = (output['user_age_range'] != "N/A") + (output['user_sex'] != "N/A") #+ (output['user_location'] != "N/A")
                    if (user_info_condition >= 2) or ((output['user_location'] != "N/A") and nb_user_tags > 0):# or nb_user_tags > 0:
                        user_info = list(output.values())

                        # Review part
                        review_title = element.find_element(by=By.XPATH, value=".//span[@class='noQuotes']").text    # Note: a xpath follows the syntax: //tagname[@attributename='value']
                        review_date = element.find_element(by=By.XPATH, value=".//span[contains(@class, 'ratingDate')]").get_attribute("title")    # Note: a xpath follows the syntax: //tagname[@attributename='value']
                        review_rating = element.find_element(by=By.XPATH, value=".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]    # Note: a xpath follows the syntax: //tagname[@attributename='value']
                        review = element.find_element(by=By.XPATH, value=".//p[@class='partial_entry']").text.replace("\n", " ")    # Note: a xpath follows the syntax: //tagname[@attributename='value']

                        user_id_element = element.find_element(by=By.XPATH, value=".//div[@class='info_text pointer_cursor']")
                        user_id = user_id_element.text.split("\n")[0]
                        user_id_link_element = user_info_element.find_element(by=By.TAG_NAME, value="a")
                        user_id_link = user_id_link_element.get_attribute("href")
                        user_id_hash = base64.b64encode(hashlib.md5(bytes(user_id, 'utf-8')).digest())
                        
                        user_info = [user_id_hash, user_id_link, user_id] + user_info
                        print("Full user info: ", user_info)
                        
                        # Write the user profile (UP) (and the review's info for the review we found the UP) in the csv file
                        all_user_info = user_info + [user_tags]
                        review_info = [restaurant_url, review_date, scraping_state['city_resume'], scraping_state['language_to_scrape'], float(review_rating)/10, review_title, review]
                        data_writer.writerow(all_user_info + review_info)

                        scraping_state["total_restaurant_users_added"] += 1


                        print(f"Obtained review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {scraping_state['city_resume']}!")
                        print("Hence added one more UP with a review! Total number of UPs added for this restaurant: ", scraping_state["total_restaurant_users_added"])
                        
                    else:
                        print(f"Skipped review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {scraping_state['city_resume']} due to lack of personal info")

                    # close the user overlay before accessing the next user's review
                    ui_backdrop = driver.find_element(by=By.XPATH, value=".//div[@class='ui_backdrop']")   # user overlay to make the user info pop up appear on the same page
                    action = webdriver.common.action_chains.ActionChains(driver)
                    action.move_to_element_with_offset(ui_backdrop, ui_backdrop.rect["width"] // (-2) + 5, 0)   # to click outside of the overlay
                    action.click()
                    action.perform()
                    
                    time.sleep(3)
                
                # change the page
                try:
                    next_page_button = driver.find_element(by=By.XPATH, value=".//a[@class='nav next ui_button primary'][normalize-space()='Next']")
                    driver.execute_script(
                    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                    next_page_button)
                    next_page_button.click()
                    time.sleep(2)
                except NoSuchElementException:  # stop iterating over the pages (as there is no more) and go to the next restaurant if possible; TODO: however, I think that there is a problem in TA website as sometimes the exception is not caught despite not being any more pages
                    break
                
            # we are no more in resume scraping mode as we already completed all the pages of the restaurant for which the previous scraping process got interrupted
            is_resume_scraping = False

        scraping_state["nb_of_cities_to_scrape"] -= 1
    
    # except Exception as e:  # TODO: see for the right way to handle any issue within the try block
    #     print("Exception occurred: ", e)
        
    #     return scraping_state
    
    return scraping_state


if __name__ == "__main__":
    
    # start time when running the script
    start_time = time.time()

    # load the chrome driver with options
    chrome_driver_path = "C:/Users/klimm/Documents/ETHZ/Semester Project/DataScraping/chromedriver_win32/chromedriver.exe"  # path to the chromedriver	
    chrome_options = Options()
    user_agent = ''.join(random.choices(string.ascii_lowercase, k=20))  # random user agent name
    chrome_options.add_argument(f'user-agent={user_agent}')
    # chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument('--load-extension=extension_3_4_4_0.crx')
    chrome_options.add_extension('./chromedriver_win32/istilldontcareaboutcookies-chrome-1.1.1_0.crx')     # to get a crx to load: https://techpp.com/2022/08/22/how-to-download-and-save-chrome-extension-as-crx/
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    # chrome_options.add_argument(r"--user-data-dir=/Users/klimm/AppData/Local/Google/Chrome/User Data") 
    # chrome_options.add_argument(r'--profile-directory=Default')
    # chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--headless")     # run the script without having a browser window open
    driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)  # creates a web driver; general variable (will not be passed to a function)
    # driver.maximize_window()

    # get from the command line the scraping hyper-parameters (e.g., language) for scraping
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--language", type=str, default="en", help="language to scrape for (e.g., en, fr, pt, es, it, de, zhCN, etc.)")
    command_line_parser.add_argument("--nb_cities", type=int, default=3, help="number of cities to scrape in a language")
    command_line_parser.add_argument("--nb_restaurants", type=int, default=3, help="number of restaurants to scrape per city")
    command_line_parser.add_argument("--nb_pages", type=int, default=3, help="number of pages to scrape per restaurant")
    command_line_parser.add_argument("--is_resume_scraping", default=False, action="store_true", help="whether to resume scraping or start a new run")

    args = command_line_parser.parse_args()

    language_to_scrape = args.language
    nb_of_cities_to_scrape = args.nb_cities
    nb_of_restaurants_to_scrape_per_city = args.nb_restaurants
    nb_of_pages_to_scrape_per_restaurant = args.nb_pages
    is_resume_scraping = args.is_resume_scraping

    # get the list of selected scrapable cities
    with open("./TA_cities.txt", "r") as file:
        cities_to_scrape = [line.rstrip() for line in file]

    # path to file to store data
    path_to_data_file = "./Data/TA_data_" + language_to_scrape + ".csv"

    # open the file to save the data
    csv_file = open(path_to_data_file, 'a', encoding="utf-8")
    data_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")

    # to keep track of the scraping process
    scraping_state = {
        "url_main": "https://www.tripadvisor.com",  # main url to scrape from
        "language_to_scrape": language_to_scrape,   # the language of the text data we are scraping
        "total_restaurant_users_added": 0,          # total number of users scraped in the previous scraping run plus the resumed scraping run
        "webpage_resume": None,                     # last url webpage on which we were scraping when the scraping was interrupted
        "city_resume": cities_to_scrape[0],           # the current city being scraped; the city that was being scraped when the scraping was interrupted
        "cities_to_scrape": cities_to_scrape,         # cities remaining to be scraped 
        "restaurants_urls_resume": [],              # restaurants remaining to be scraped in the current city
        "page_nb_resume": 1,                        # the page number for the restaurant that was being scraped when the scraping was interrupted; 1 by default as we loop from 0 with a -1 shift
        "nb_of_cities_to_scrape": nb_of_cities_to_scrape,   # total number of cities that are meant to be scraped (since the start of the initial scraping run)
        "nb_of_restaurants_to_scrape_per_city": nb_of_restaurants_to_scrape_per_city,   # total number of restaurants per city that are meant to be scraped
        "nb_of_pages_to_scrape_per_restaurant": nb_of_pages_to_scrape_per_restaurant,   # total number of pages per restaurant that are meant to be scraped
    }

    # print("\n\n", scraping_state)

    basic_user_info_header = ["user_name", "user_ta_level", "user_age_range", "user_sex", "user_location", "user_nb_contributions", "user_nb_cities_visited", "user_nb_helpful_votes", "user_nb_photos"]
    id_user_info_header = ["user_id_hash", "user_id_link", "user_id"]
    additional_user_info_header = ["user_tags"]
    restaurant_review_header = ["restaurant_reviewed_url", "review_date", "review_city", "review_lang", "review_rating", "review_title", "review"]
    full_row_header = id_user_info_header + basic_user_info_header + additional_user_info_header + restaurant_review_header

    if is_resume_scraping:
        
        # load the pickled saved dictionary containing the scraping state when the scraping got interrupted; hence rewrite the initial state
        with open(f'parameters_resume_scraping_in_{language_to_scrape}.txt', 'rb') as f:
            scraping_state = pickle.loads(f.read())
        
        print(f"Resuming scraping from webpage url {scraping_state['webpage_resume']} for language {scraping_state['language_to_scrape']}, city {scraping_state['city_resume']} at page {scraping_state['page_nb_resume']}.")

    else:
        # write header of the csv file
        data_writer.writerow(full_row_header)
        
        print(f"""Starting a completely new scraping run from main webpage url {scraping_state['url_main']} for language {scraping_state['language_to_scrape']}. 
              Going to scrape {nb_of_cities_to_scrape} cities, {nb_of_restaurants_to_scrape_per_city} restaurants per city and a maximum of {nb_of_pages_to_scrape_per_restaurant} pages per restaurant.""")


    # start scraping
    scraping_output = start_scraping(is_resume_scraping, scraping_state, data_writer)

    print(scraping_output)

    # save the scraping state in case the scraping got interrupted
    with open(f'parameters_resume_scraping_in_{language_to_scrape}.txt', 'wb') as f:
        pickle.dump(scraping_output, f)

 
    # close csv file as nothing more to write for now
    csv_file.close()

    # finish properly with the driver
    driver.close()
    driver.quit()

    # clean the dataset (i.e., remove duplicates)
    dataset = pd.read_csv(f'./Data/TA_data_{language_to_scrape}.csv', delimiter="\t", encoding="utf-8", )
    print("Shape of uncleaned dataset: ", dataset.shape)
    clean_dataset = dataset.drop_duplicates(keep=False)
    print("Shape of cleaned dataset: ", clean_dataset.shape)
    print('Cleaned dataset (simply dropped duplicates created by potential scraping issues):\n', clean_dataset)

    # convert the pandas dataframe to a CSV file
    clean_dataset.to_csv(f'./Data/TA_cleaned_data_{language_to_scrape}.csv', sep="\t", encoding="utf-8", index=False)
    
    # convert the pandas dataframe to a XML file
    clean_dataset.to_xml(f'./Data/TA_cleaned_data_{language_to_scrape}.xml')
    # with open(f'./Data/TA_cleaned_data_{language_to_scrape}.xml', 'w') as file: 
    #     file.write(clean_dataset.to_xml())

    # time when end of scraping
    end_time = time.time()
    print("Finished scraping TripAdvisor")
    print("Time elapsed for the scraping run: ", int(end_time - start_time) // 60, " minutes and ", int(end_time - start_time) % 60, " seconds")
