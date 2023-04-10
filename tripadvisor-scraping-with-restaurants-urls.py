import TA_utility

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


def start_scraping(driver, data_writer, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=0):
    
    restaurants_data = []
    
    # open file TA_restaurants_urls.csv to get the list of restaurants urls to scrape
    with open('./Data/TA_restaurants_urls.csv', 'r') as f:
        restaurants_data_str = f.read().splitlines()
        for data in restaurants_data_str[1:]:   # skip the header
            restaurants_data.append(data.split('\t'))

    wait_driver = WebDriverWait(driver, 20)

    try:

        # iterate over the restaurants (urls) to scrape from the file TA_restaurants_urls.csv
        for restaurant_index, (city, restaurant_url) in enumerate(restaurants_data[restaurant_url_index:]):  # only select the first 3 top restaurants for now
        
            print("Currently scraping restaurant: ", restaurant_url)

            driver.get(restaurant_url)
        
            time.sleep(3)
            
            # scroll down to reach the general reviews panel
            # dummy_reviews_point_for_scroll = driver.find_element(by=By.XPATH, value="//div[@class='title_text']")
            dummy_reviews_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, "//div[@class='title_text']")))

            driver.execute_script(
            "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
            dummy_reviews_point_for_scroll)

            try: 
                # remove the sticky bar to avoid clicking on it
                driver.execute_script("""
                var l = document.getElementsByClassName("stickyBar")[0];
                l.parentNode.removeChild(l);
                """)
            except NoSuchElementException:
                print("Problem removing the sticky bar")

            # Select the language of the reviews
            language_id_selector = "filters_detail_language_filterLang_" + language_to_scrape

            try:
                # click on the wanted language in the list of languages directly available
                time.sleep(1)
                selected_lang = driver.find_element(by=By.XPATH, value=f'.//div[@class="item"][@data-value={language_to_scrape}]/label[@class="label container"][@for={language_id_selector}]')
                # selected_lang = wait_driver.until(EC.element_to_be_clickable((By.XPATH, f'.//div[@class="item"][@data-value={language_to_scrape}]/label[@class="label container"][@for={language_id_selector}]')))
                time.sleep(1)
                selected_lang.click() 
            except NoSuchElementException:
                try:
                    # click on the "More languages" button to get the overlay with more language choices
                    more_languages_bar = driver.find_element(by=By.XPATH, value='.//div[@class="taLnk"]')
                    # more_languages_bar = wait_driver.until(EC.element_to_be_clickable((By.XPATH, f'.//div[@class="taLnk"]')))
                    more_languages_bar.click()
                    time.sleep(2)
                    # click on the wanted language in the "More languages" overlay
                    language_in_bar = driver.find_element(by=By.ID, value=language_id_selector)
                    # language_in_bar = wait_driver.until(EC.element_to_be_clickable((By.ID, language_id_selector)))

                    driver.execute_script(
                        "arguments[0].click();",
                        language_in_bar)
                except NoSuchElementException:  # go to the next restaurant              
                    continue

            # make sure to scroll down before getting the reviews
            try:
                dummy_showreviews_point_for_scroll = driver.find_element(by=By.XPATH, value="//div[normalize-space()='Show reviews that mention']")
                # dummy_showreviews_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, "//div[normalize-space()='Show reviews that mention']")))
                
                driver.execute_script(
                "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                dummy_showreviews_point_for_scroll)
            except NoSuchElementException:
                try:
                    dummy_firstreview_point_for_scroll = driver.find_element(by=By.XPATH, value=".//div[@class='review-container']")
                    # dummy_firstreview_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='review-container']")))

                    driver.execute_script(
                    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                    dummy_firstreview_point_for_scroll)
                except NoSuchElementException:
                    print("No reviews found for this restaurant")
                    continue    # go to the next restaurant
            
            # iterate over the restaurant's pages
            for page_index in range(0, nb_of_pages_to_scrape_per_restaurant):

                time.sleep(3)

                wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='review-container']")))
                reviews_container = driver.find_elements(by=By.XPATH, value=".//div[@class='review-container']")

                print(f"{len(reviews_container)} reviews found on page {page_index+1}")
                
                for review_index, element in enumerate(reviews_container):   # a container element is a user review
                    print(f"Entered in review {review_index+1} from page {page_index+1}")

                    # If the user review is a review translated from English to another language (e.g., zhCH), then skip it (TODO: see if still need to collect it or not)
                    # if not element.find_elements(by=By.XPATH, value="//div[@data-prwidget-name='reviews_google_translate_button_hsx']"):    # using xpath to find the element. find_elements returns empty list (which is equal to False)
                    #     continue

                    # Open all the reviews "More..." on the page to get the full review
                    wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//span[@class='taLnk ulBlueLinks']")))
                    review_expander_buttons = element.find_elements(by=By.XPATH, value="//span[@class='taLnk ulBlueLinks']")
                    time.sleep(1)
                    if review_expander_buttons:
                        review_expander_buttons[0].click()
                        time.sleep(1)  

                    # a user may have a deactivated account, in which case the user info overlay pop up will not appear
                    try:
                        overlay_element = element.find_element(by=By.XPATH, value=".//div[@class='memberOverlayLink clickable']")    # user overlay to make the user info pop up appear on the same page
                        overlay_element.click()

                    except NoSuchElementException:
                        print("No user info found for this review. Account has been deleted.")
                        continue

                    # user_overlay_info = driver.find_element(by=By.XPATH, value=".//div[@class='memberOverlay simple container moRedesign']") 
                    user_overlay_info = wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='memberOverlay simple container moRedesign']")))
                    user_info_element = user_overlay_info.find_element(by=By.XPATH, value=".//div[@class='memberOverlayRedesign g10n']")
                    user_info = user_info_element.text.split("\n")        # get the user info   # E.g: ['UserName', 'Level 4 Contributor', 'Send Message', 'Tripadvisor member since 2007', '35-49 man from Wednesbury, United Kingdom', ' 37 Contributions', ' 38 Cities visited', ' 28 Helpful votes', ' 14 Photos', 'Like a Local Thrill Seeker Shopping Fanatic Peace and Quiet Seeker', 'View all', 'REVIEW DISTRIBUTION', 'Excellent\t', '\t32', 'Very good\t', '\t2', 'Average\t', '\t2', 'Poor\t', '\t0', 'Terrible\t', '\t1']
                    
                    # Parse the user info according to the hardcoded keywords in the list
                    parsed_user_info = parse_user_info(user_info)

                    # Get the user tags
                    user_tags = user_overlay_info.find_elements(by=By.XPATH, value=".//a[@class='memberTagReviewEnhancements']")
                    nb_user_tags = len(user_tags)
                    user_tags = ";".join([tag.text for tag in user_tags]) if nb_user_tags else "N/A"  # we arbritrarily define semicolon as the separator within the feature user_tags
                    
                    # If the UP is complete enough; TODO: see what can be a good enough condition (e.g., if we should also use the number of tags as part of the condition although it may be too restrictive)
                    user_info_condition = (parsed_user_info['user_age_range'] != "N/A") + (parsed_user_info['user_sex'] != "N/A") #+ (parsed_user_info['user_location'] != "N/A")
                    if (user_info_condition >= 2) or ((parsed_user_info['user_location'] != "N/A") and nb_user_tags > 0):# or nb_user_tags > 0:
                        user_info = list(parsed_user_info.values())

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
                        review_info = [restaurant_url, review_date, city, language_to_scrape, float(review_rating)/10, review_title, review]
                        data_writer.writerow(all_user_info + review_info)

                        print(f"Obtained review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {city}!")
                        print("Hence added one more UP with a review!")
                        
                    else:
                        print(f"Skipped review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {city} due to lack of personal info")

                    # close the user overlay before accessing the next user's review
                    # ui_backdrop = driver.find_element(by=By.XPATH, value=".//div[@class='ui_backdrop']")   # user overlay to make the user info pop up appear on the same page
                    ui_backdrop = wait_driver.until(EC.element_to_be_clickable((By.XPATH, ".//div[@class='ui_backdrop']")))
                    action = webdriver.common.action_chains.ActionChains(driver)
                    action.move_to_element_with_offset(ui_backdrop, ui_backdrop.rect["width"] // (-2) + 5, 0)   # to click outside of the overlay
                    action.click()
                    action.perform()
                    time.sleep(2)
                
                # change the page
                try:
                    next_page_button = driver.find_element(by=By.XPATH, value=".//a[@class='nav next ui_button primary'][normalize-space()='Next']")
                    # next_page_button = wait_driver.until(EC.element_to_be_clickable((By.XPATH, ".//a[@class='nav next ui_button primary'][normalize-space()='Next']")))
                    driver.execute_script(
                    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
                    next_page_button)
                    next_page_button.click()
                    time.sleep(2)
                except NoSuchElementException:  # stop iterating over the pages (as there is no more) and go to the next restaurant if possible; TODO: however, I think that there is a problem in TA website as sometimes the exception is not caught despite not being any more pages
                    break
        
    except Exception as e:
        print("Exception occured: ", e)
        print("Exception occured in restaurant: ", restaurant_url)
        print("Exception occured in city: ", city)
        print("Restaurant index: ", restaurant_index)

        # resume scraping if an error occured during scraping
        # start_scraping(data_writer, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=restaurant_index)
        return restaurant_index

    return restaurant_index


if __name__ == "__main__":
    
    # start time when running the script
    start_time = time.time()

    # get the driver
    driver = TA_utility.get_driver()

    # get from the command line the scraping hyper-parameters (e.g., language) for scraping
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--language", type=str, default="en", help="language to scrape for (e.g., en, fr, pt, es, it, de, zhCN, etc.)")
    command_line_parser.add_argument("--nb_pages", type=int, default=3, help="number of pages to scrape per restaurant")
    command_line_parser.add_argument("--resume_restaurant_url_index", type=int, default=0, help=" ")

    args = command_line_parser.parse_args()

    language_to_scrape = args.language
    nb_of_pages_to_scrape_per_restaurant = args.nb_pages
    resume_restaurant_url_index = args.resume_restaurant_url_index


    # get the list of selected scrapable cities
    with open("./TA_cities.txt", "r") as file:
        cities_to_scrape = [line.rstrip() for line in file]

    # path to file to store data
    path_to_data_file = "./Data/TA_data_" + language_to_scrape + ".csv"

    # open the file to save the data
    csv_file = open(path_to_data_file, 'a', encoding="utf-8")
    data_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")

    basic_user_info_header = ["user_name", "user_ta_level", "user_age_range", "user_sex", "user_location", "user_nb_contributions", "user_nb_cities_visited", "user_nb_helpful_votes", "user_nb_photos"]
    id_user_info_header = ["user_id_hash", "user_id_link", "user_id"]
    additional_user_info_header = ["user_tags"]
    restaurant_review_header = ["restaurant_reviewed_url", "review_date", "review_city", "review_lang", "review_rating", "review_title", "review"]
    full_row_header = id_user_info_header + basic_user_info_header + additional_user_info_header + restaurant_review_header

    if resume_restaurant_url_index == 0:
        # write header of the csv file
        data_writer.writerow(full_row_header)
    
    # start scraping
    scraping_output = start_scraping(driver, data_writer, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=resume_restaurant_url_index)

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
    # clean_dataset.to_xml(f'./Data/TA_cleaned_data_{language_to_scrape}.xml')
    # with open(f'./Data/TA_cleaned_data_{language_to_scrape}.xml', 'w') as file: 
    #     file.write(clean_dataset.to_xml())

    # time when end of scraping
    end_time = time.time()
    print("Finished scraping TripAdvisor")
    print("Time elapsed for the scraping run: ", int(end_time - start_time) // 60, " minutes and ", int(end_time - start_time) % 60, " seconds")
