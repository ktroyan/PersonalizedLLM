import TA_utility

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException, NoSuchWindowException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import os
from os.path import exists
import time
import csv
import argparse
import hashlib
import base64
import re
from collections import OrderedDict

def collect_restaurants_urls():
    print("Currently collecting restaurants to scrape...")
    with open("tripadvisor_save_restaurants_urls.py") as f:
        exec(f.read())

def get_list_of_restaurants_data():
    restaurants_data = []
    with open('./TripAdvisor/Data/TA_restaurants_urls.csv', 'r') as f:
        restaurants_data_str = f.read().splitlines()
        for data in restaurants_data_str[1:]:   # skip the header
            restaurants_data.append(data.split('\t'))
    return restaurants_data

def open_restaurant_webpage(restaurant_url):
    driver.get(restaurant_url)

def scroll_to_reviews_panel(driver, wait_driver):
    # scroll down to reach the general reviews panel
    dummy_reviews_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, "//div[@class='title_text']")))
    driver.execute_script(
    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
    dummy_reviews_point_for_scroll)

def remove_sticky_bar(driver):
    try:
        # TODO: Improve sticky bar handling
        # remove the sticky bar to avoid clicking on it
        # wait_driver.until(EC.visibility_of_element_located((By.XPATH, "//div[normalize-space()='Show reviews that mention']")))
        driver.execute_script("""
        var l = document.getElementsByClassName("stickyBar")[0];
        if (l && l.parentNode) {
            l.parentNode.removeChild(l);}
        """)
        # driver.execute_script("""
        # var l = document.getElementsByClassName("stickyBar")[0];
        # l.parentNode.removeChild(l);
        # """)
    except NoSuchElementException:
        print("Problem removing the sticky bar")

def select_reviews_language(driver, language_to_scrape, language_id_selector):
    # click on the wanted language in the list of languages directly available
    time.sleep(1)
    selected_lang = driver.find_element(by=By.XPATH, value=f'.//div[@class="item"][@data-value={language_to_scrape}]/label[@class="label container"][@for={language_id_selector}]')
    # selected_lang = wait_driver.until(EC.element_to_be_clickable((By.XPATH, f'.//div[@class="item"][@data-value={language_to_scrape}]/label[@class="label container"][@for={language_id_selector}]')))
    time.sleep(1)
    selected_lang.click() 

def click_more_and_select_reviews_language(driver, language_id_selector):
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

def scroll_to_reviews(driver):
    try:
        dummy_showreviews_point_for_scroll = driver.find_element(by=By.XPATH, value="//div[normalize-space()='Show reviews that mention']")
        # dummy_showreviews_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, "//div[normalize-space()='Show reviews that mention']")))
        
        driver.execute_script(
        "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
        dummy_showreviews_point_for_scroll)
    except NoSuchElementException:
        dummy_firstreview_point_for_scroll = driver.find_element(by=By.XPATH, value=".//div[@class='review-container']")
        # dummy_firstreview_point_for_scroll = wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='review-container']")))

        driver.execute_script(
        "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
        dummy_firstreview_point_for_scroll)

def get_reviews_container(driver, wait_driver):
    wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='review-container']")))
    reviews_container = driver.find_elements(by=By.XPATH, value=".//div[@class='review-container']")
    return reviews_container

def skip_translated_review(element):
    return not element.find_elements(by=By.XPATH, value="//div[@data-prwidget-name='reviews_google_translate_button_hsx']")     # using xpath to find the element. find_elements returns empty list (which is equal to False)

def expand_long_text_reviews(driver, wait_driver, element):
    # wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//span[@class='taLnk ulBlueLinks']")))
    # review_expander_buttons = element.find_elements(by=By.XPATH, value="//span[@class='taLnk ulBlueLinks']")
    
    # time.sleep(1)
    # if review_expander_buttons:
    #     review_expander_buttons[0].click()
    #     time.sleep(1)  

    review_expander_button = element.find_element(by=By.XPATH, value="//span[@class='taLnk ulBlueLinks']")
    driver.execute_script("arguments[0].click();", review_expander_button) 
    time.sleep(1)

def open_user_overlay(element):
    overlay_element = element.find_element(by=By.XPATH, value=".//div[@class='memberOverlayLink clickable']")    # user overlay to make the user info pop up appear on the same page
    overlay_element.click()

def get_user_info(wait_driver):
    # user_overlay_info = driver.find_element(by=By.XPATH, value=".//div[@class='memberOverlay simple container moRedesign']") 
    user_overlay_info = wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='memberOverlay simple container moRedesign']")))
    user_info_element = user_overlay_info.find_element(by=By.XPATH, value=".//div[@class='memberOverlayRedesign g10n']")
    user_info = user_info_element.text.split("\n")        # get the user info   # E.g: ['UserName', 'Level 4 Contributor', 'Send Message', 'Tripadvisor member since 2007', '35-49 man from Wednesbury, United Kingdom', ' 37 Contributions', ' 38 Cities visited', ' 28 Helpful votes', ' 14 Photos', 'Like a Local Thrill Seeker Shopping Fanatic Peace and Quiet Seeker', 'View all', 'REVIEW DISTRIBUTION', 'Excellent\t', '\t32', 'Very good\t', '\t2', 'Average\t', '\t2', 'Poor\t', '\t0', 'Terrible\t', '\t1']
    return user_info, user_info_element, user_overlay_info

# define a matchers dict and a processors dict to get the user info in the user overlay through RegEx; note that the parentheses define the part taken
matchers = {    
    "user_ta_level": lambda item: re.search("Level ([0-9]+)", item),
    "user_age_range": lambda item: re.search("[0-9][0-9]-[0-9][0-9]", item) or re.search("[0-9][0-9]\+", item),
    "user_sex": lambda item: re.search("woman", item) or re.search("man", item),
    "user_location": lambda item: re.search("[fF]rom (.*)", item),
    "user_nb_contributions": lambda item: re.search("([0-9]+) Contributions", item),
    "user_nb_cities_visited": lambda item: re.search("([0-9]+) Cities visited", item),
    "user_nb_helpful_votes": lambda item: re.search("([0-9]+) Helpful votes", item),
    "user_nb_photos": lambda item: re.search("([0-9]+) Photos", item)
}

def get_match(index=0):
    return lambda match: match.group(index)

processors = {  # argument 1 means we take the group between parentheses of the RegEx, 0 means we take the whole match
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
            if output[key] != 'N/A':
                continue
            
            m = matcher(item)   # match the item with the RegEx
            if m:   # if the item matched the RegEx
                output[key] = processors[key](m)
            
    return output

def get_user_tags(user_overlay_info):
    user_tags = user_overlay_info.find_elements(by=By.XPATH, value=".//a[@class='memberTagReviewEnhancements']")
    nb_user_tags = len(user_tags)
    user_tags = ";".join([tag.text for tag in user_tags]) if nb_user_tags else "N/A"  # we arbritrarily define semicolon as the separator within the feature user_tags
    return user_tags, nb_user_tags

def define_collection_condition(parsed_user_info, nb_user_tags):
    # If the UP is complete enough; TODO: see what can be a good enough condition (e.g., if we should also use the number of tags as part of the condition although it may be too restrictive)
    user_info_condition = (parsed_user_info['user_age_range'] != "N/A") + (parsed_user_info['user_sex'] != "N/A") #+ (parsed_user_info['user_location'] != "N/A")
    return (user_info_condition >= 2) or ((parsed_user_info['user_location'] != "N/A") and nb_user_tags > 0) # or nb_user_tags > 0:

def get_review_elements(element):
    review_title = element.find_element(by=By.XPATH, value=".//span[@class='noQuotes']").text    # Note: a xpath follows the syntax: //tagname[@attributename='value']
    review_date = element.find_element(by=By.XPATH, value=".//span[contains(@class, 'ratingDate')]").get_attribute("title")    # Note: a xpath follows the syntax: //tagname[@attributename='value']
    review_rating = element.find_element(by=By.XPATH, value=".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]    # Note: a xpath follows the syntax: //tagname[@attributename='value']
    review = element.find_element(by=By.XPATH, value=".//p[@class='partial_entry']").text.replace("\n", " ")    # Note: a xpath follows the syntax: //tagname[@attributename='value']
    return review_title, review_date, review_rating, review

def collect_sample(data_writer, city, restaurant_url, parsed_user_info, user_tags, user_info_element, element):
    user_info = list(parsed_user_info.values())

    # Review part
    review_title, review_date, review_rating, review = get_review_elements(element)

    user_id_element = element.find_element(by=By.XPATH, value=".//div[@class='info_text pointer_cursor']")
    user_id = user_id_element.text.split("\n")[0]
    user_id_link_element = user_info_element.find_element(by=By.TAG_NAME, value="a")
    user_id_link = user_id_link_element.get_attribute("href")
    user_id_hash = base64.b64encode(hashlib.md5(bytes(user_id, 'utf-8')).digest())
    
    user_info = [user_id_hash, user_id_link, user_id] + user_info
    
    all_user_info = user_info + [user_tags]
    print("User info: ", all_user_info)

    review_info = [restaurant_url, review_date, city, language_to_scrape, float(review_rating)/10, review_title, review]

    # Write the user profile (UP) (and the review's info for the review we found the UP) in the csv file
    data_writer.writerow(all_user_info + review_info)
     
def close_user_overlay(driver, wait_driver):
    # close the user overlay before accessing the next user's review
    # ui_backdrop = driver.find_element(by=By.XPATH, value=".//div[@class='ui_backdrop']")   # user overlay to make the user info pop up appear on the same page
    ui_backdrop = wait_driver.until(EC.element_to_be_clickable((By.XPATH, ".//div[@class='ui_backdrop']")))
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element_with_offset(ui_backdrop, ui_backdrop.rect["width"] // (-2) + 5, 0)   # to click outside of the overlay
    action.click()
    action.perform()
    time.sleep(2)

def go_to_next_page(driver):
    next_page_button = driver.find_element(by=By.XPATH, value=".//a[@class='nav next ui_button primary'][normalize-space()='Next']")
    # next_page_button = wait_driver.until(EC.element_to_be_clickable((By.XPATH, ".//a[@class='nav next ui_button primary'][normalize-space()='Next']")))
    driver.execute_script(
    "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
    next_page_button)
    next_page_button.click()
    time.sleep(2)

def rewrite_restaurants_urls_file(path_to_restaurant_file, restaurant_full_row_header, restaurants_data):
    # open file TA_restaurants_urls.csv and update it (by overwriting it completely, but with new data)
    with open(path_to_restaurant_file, 'w') as restaurant_csv_file:
        restaurant_file_update_writer = csv.writer(restaurant_csv_file, delimiter="\t", lineterminator="\n")
        restaurant_file_update_writer.writerow(restaurant_full_row_header)
        for data in restaurants_data:
            restaurant_file_update_writer.writerow(data)

# this function goes over a list of restaurants and collects the user profiles and associated reviews of each restaurant in a given language if the colelction criterion is met
def start_scraping(driver, data_writer, path_to_restaurant_file, restaurant_full_row_header, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=0):
    
    print("Starting to scrape TripAdvisor...")

    # check if the file containing the urls of the restaurants to scrape exists, if not, create it by running the appropriate script
    if not("TA_restaurants_urls.csv" in os.listdir("./TripAdvisor/Data")):
        # collect the restaurants urls to scrape and write them in the file TA_restaurants_urls.csv
        collect_restaurants_urls()
    
    # open file TA_restaurants_urls.csv to get the list of restaurants urls (and associated data) to scrape
    restaurants_data = get_list_of_restaurants_data()

    wait_driver = WebDriverWait(driver, 20)

    nb_of_samples_added = 0

    restaurant_index_resume = 0

    try:

        # iterate over the restaurants (urls) to scrape from the file TA_restaurants_urls.csv
        for restaurant_index, (city, restaurant_url, scraped_languages) in enumerate(restaurants_data[restaurant_url_index:]):
            if language_to_scrape in scraped_languages.split(','):  # we already scraped this restaurant in the current scraping language
                continue

            print("Currently scraping restaurant: ", restaurant_url)

            restaurant_index_resume = restaurant_url_index + restaurant_index

            open_restaurant_webpage(restaurant_url)
        
            time.sleep(3)
            
            # Scroll down to the reviews panel
            scroll_to_reviews_panel(driver, wait_driver)

            remove_sticky_bar(driver)

            # Select the language of the reviews
            language_id_selector = "filters_detail_language_filterLang_" + language_to_scrape
            try:
                select_reviews_language(driver, language_to_scrape, language_id_selector)
            except NoSuchElementException:
                try:
                    click_more_and_select_reviews_language(driver, language_id_selector)
                except NoSuchElementException:  # go to the next restaurant
                    restaurants_data[restaurant_index_resume][2] += f',{language_to_scrape}'    # consider this restaurant as scraped     
                    print("Going to the next restaurant as language could not be selected for this restaurant.")         
                    continue

            # Make sure to scroll down to the reviews before getting the reviews
            try:
                scroll_to_reviews(driver)
            except NoSuchElementException:
                print("No reviews found for this restaurant")
                restaurants_data[restaurant_index_resume][2] += f',{language_to_scrape}'    # consider this restaurant as scraped
                print("Going to the next restaurant as scroll to the reviews failed for this restaurant.")
                continue    # go to the next restaurant
        
            # iterate over the restaurant's pages
            for page_index in range(0, nb_of_pages_to_scrape_per_restaurant):

                time.sleep(3)

                reviews_container = get_reviews_container(driver, wait_driver)

                print(f"{len(reviews_container)} reviews found on page {page_index+1}")
                
                for review_index, element in enumerate(reviews_container):   # a container element is a user review
                    print(f"Entered in review {review_index+1} from page {page_index+1}")

                    # if the user review is a review translated from English to another language (e.g., zhCH), then skip it; TODO: choose if still need to collect it or not
                    # if skip_translated_review(element):    
                    #     continue

                    # Expand/Open the review with "More..." on the page to get the full review
                    try:
                        expand_long_text_reviews(driver, wait_driver, element)
                    except TimeoutException:
                        print("'More' button not found for this review")
                        continue
                    except NoSuchElementException:
                        print("'More' button not found for this review")
                        continue
                    except ElementClickInterceptedException:
                        print("'More' button hidden for this review")
                        continue


                    # a user may have a deactivated account, in which case the user info overlay pop up will not appear
                    try:
                        open_user_overlay(element)
                    except NoSuchElementException:
                        print("No user info found for this review. Account has been deleted.")
                        continue

                    # Get the user info
                    user_info, user_info_element, user_overlay_info = get_user_info(wait_driver)

                    # Parse the user info according to the hardcoded keywords in the list
                    parsed_user_info = parse_user_info(user_info)

                    # Get the user tags
                    user_tags, nb_user_tags = get_user_tags(user_overlay_info)
                    
                    # Get the condition for the collection of the data sample
                    collection_condition = define_collection_condition(parsed_user_info, nb_user_tags)

                    if collection_condition:
                        collect_sample(data_writer, city, restaurant_url, parsed_user_info, user_tags, user_info_element, element)
                        nb_of_samples_added += 1
                        print(f"Obtained review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {city}!")
                        print(f"Hence added one more UP with a review for a total of {nb_of_samples_added} samples currently added for this run!")
                        
                    else:
                        print(f"Skipped review-user {review_index+1} of page {page_index+1} for restaurant {restaurant_url} in city {city} due to lack of personal info")

                    close_user_overlay(driver, wait_driver)
                
                # go to the next reviews page of the restaurant
                try:
                    go_to_next_page(driver)
                except NoSuchElementException:  # stop iterating over the pages (as there is no more) and go to the next restaurant if possible
                    break

            restaurants_data[restaurant_index_resume][2] += f',{language_to_scrape}'

    except (NoSuchWindowException, KeyboardInterrupt) as e:
        print("Exception occured: ", e)
        print("Exception occured in restaurant: ", restaurant_url)
        print("Exception occured in city: ", city)
        print("Iterated restaurant index: ", restaurant_index)
        print("Restaurant index when exception occurred during scraping run: ", restaurant_index_resume)
        
        print("\n\nNOT resuming scraping as the browser was closed (voluntary or not).\n")

        rewrite_restaurants_urls_file(path_to_restaurant_file, restaurant_full_row_header, restaurants_data)
        
        print("New restaurant index to resume scraping (same as where it got interrupted): ", restaurant_index_resume, "\n")

        # save the scraping state in case the scraping got interrupted
        with open(f'./TripAdvisor/parameters_resume_scraping_in_{language_to_scrape}.txt', 'w') as f:
            f.write(str(restaurant_index_resume))
        
        return restaurant_index_resume  # restaurant index (in the file) at which the scraping can be resumed (i.e., one after the index in the file where the scraping run failed)

    except Exception as e:
        print("Exception occured: ", e)
        print("Exception occured in restaurant: ", restaurant_url)
        print("Exception occured in city: ", city)
        print("Iterated restaurant index: ", restaurant_index)
        print("Restaurant index when exception occurred during scraping run: ", restaurant_index_resume)
        
        # A new exception occured, so we mark the restaurant as scraped in the file and continue to the next restaurant
        restaurants_data[restaurant_index_resume][2] += f',{language_to_scrape}'
        rewrite_restaurants_urls_file(path_to_restaurant_file, restaurant_full_row_header, restaurants_data)

        # Resume scraping if an error different than closing the driver occured during scraping
        print("\n\nResuming scraping because an error occurred...\n\n")
        restaurant_index_resume += 1
        print("New restaurant index to resume scraping: ", restaurant_index_resume)
        scraping_output = start_scraping(driver, data_writer, path_to_restaurant_file, restaurant_full_row_header, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=restaurant_index_resume)  # note that we pass the restaurant index at which to  resume the scraping run (see the restaurant for-loop)
        
        # save the scraping state in case the scraping got interrupted
        with open(f'./TripAdvisor/parameters_resume_scraping_in_{language_to_scrape}.txt', 'w') as f:
            f.write(str(scraping_output))
        
        return restaurant_index_resume  # restaurant index (in the file) at which the scraping can be resumed (i.e., one after the index in the file where the scraping run failed)
    
    rewrite_restaurants_urls_file(path_to_restaurant_file, restaurant_full_row_header, restaurants_data)
                
    restaurant_index_resume += 1
    print("Complete scraping run finished successfully. New restaurant index after last scraping run: ", restaurant_index_resume)
    return restaurant_index_resume


if __name__ == "__main__":
    
    # start time when running the script
    start_time = time.time()

    # get the driver
    driver = TA_utility.get_driver()

    # get from the command line the scraping hyper-parameters (e.g., language) for scraping
    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--language", type=str, default="en", help="language to scrape for (e.g., en, fr, pt, es, it, de, zhCN, etc.)")
    command_line_parser.add_argument("--nb_pages", type=int, default=3, help="number of pages to scrape per restaurant")
    command_line_parser.add_argument("--resume_restaurant_url_index", type=int, default=0, help="index of the restaurant url to resume scraping from (e.g., 0, 1, 2, etc.) if needed")

    args = command_line_parser.parse_args()

    language_to_scrape = args.language
    nb_of_pages_to_scrape_per_restaurant = args.nb_pages
    resume_restaurant_url_index = args.resume_restaurant_url_index

    # get the list of selected scrapable cities
    with open("./TripAdvisor/TA_cities.txt", "r") as file:
        cities_to_scrape = [line.rstrip() for line in file]

    # open file and create writer to save the data
    path_to_data_file = "./TripAdvisor/Data/TA_data_" + language_to_scrape + ".csv"
    data_csv_file = open(path_to_data_file, 'a', encoding="utf-8")
    data_writer = csv.writer(data_csv_file, delimiter="\t", lineterminator="\n")

    id_user_info_header = ["user_id_hash", "user_id_link", "user_id"]
    basic_user_info_header = ["user_name", "user_ta_level", "user_age_range", "user_sex", "user_location", "user_nb_contributions", "user_nb_cities_visited", "user_nb_helpful_votes", "user_nb_photos"]
    additional_user_info_header = ["user_tags"]
    restaurant_review_header = ["restaurant_reviewed_url", "review_date", "review_city", "review_lang", "review_rating", "review_title", "review"]
    data_full_row_header = id_user_info_header + basic_user_info_header + additional_user_info_header + restaurant_review_header

    # write header of the csv file if there is no header yet
    if os.stat(path_to_data_file).st_size == 0:
        data_file_has_header = False
    else:
        data_file_has_header = True

    if not(data_file_has_header):
        # write header of the csv file
        data_writer.writerow(data_full_row_header)

    # needed variables to update the TA_restaurants_urls.csv file during scraping
    path_to_restaurant_file = "./TripAdvisor/Data/TA_restaurants_urls.csv"
    restaurant_full_row_header = ["city", "restaurant_url", "scraped_lang"]
    
    # start scraping
    scraping_output = start_scraping(driver, data_writer, path_to_restaurant_file, restaurant_full_row_header, language_to_scrape, nb_of_pages_to_scrape_per_restaurant, restaurant_url_index=resume_restaurant_url_index)

    print(scraping_output)

    # save the scraping state in case the scraping got interrupted
    with open(f'./TripAdvisor/parameters_resume_scraping_in_{language_to_scrape}.txt', 'w') as f:
        f.write(str(scraping_output))
 
    # close csv files as nothing more to write for now
    data_csv_file.close()

    # finish properly with the driver
    driver.close()
    driver.quit()

    # time spent for the full scraping run
    end_time = time.time()
    print("Finished scraping TripAdvisor")
    print("Time elapsed for the scraping run: ", int(end_time - start_time) // 60, " minutes and ", int(end_time - start_time) % 60, " seconds")
