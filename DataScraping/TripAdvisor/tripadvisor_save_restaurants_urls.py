from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import time
import csv
import argparse
import random
import string

import TA_utility

def get_restaurants_urls(cities_to_scrape):
    
    print("Cities to scrape: ")
    print(cities_to_scrape)

    # save in a csv file the restaurants urls for a city
    csv_restaurants_file = open(f"./TripAdvisor/Data/TA_restaurants_urls.csv", "w", encoding="utf-8")
    csv_writer_restaurant = csv.writer(csv_restaurants_file, delimiter="\t", lineterminator="\n")
    csv_writer_restaurant.writerow(["city", "restaurant_url", "scraped_lang"])

    wait_driver = WebDriverWait(driver, 20)

    # iterate over the cities to scrape
    for city_index, review_city_name in enumerate(cities_to_scrape):

        print("\nGetting the restaurants urls in: " + review_city_name)
        
        # open the website
        driver.get("https://www.tripadvisor.com")  
        # scale_element = driver.find_element(by=By.TAG_NAME, value="body")
        # driver.execute_script("arguments[0].style.transform='scale(1)';", scale_element)


        # click on the "Restaurants" button
        restaurants_button = wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//a[@href='/Restaurants']")))
        restaurants_button.click()


        # click on the "Search" bar and input the city name
        search_city_bar = wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='slvrn Z0 Wh rsqqi EcFTp GADiy']//input[@placeholder='Where to?']")))
        search_city_bar.click()
        time.sleep(1)
        search_city_bar.send_keys(review_city_name)
        time.sleep(3)


        # click on the selected city (which is the first suggestion)
        selected_city = wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='XYHql z RJdtB']")))
        selected_city.click()

        # scroll down to reach the restaurants top reviews
        top_restaurants_title = wait_driver.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='pFMac b Cj']")))

        driver.execute_script(
        "arguments[0].scrollIntoView();",   # can add the following arg to scrollIntoView(): {behavior: 'smooth', block: 'end', inline: 'end'}
        top_restaurants_title)
        
        time.sleep(2)

        # get url of the current webpage
        print(driver.current_url)

        # since it is a new city, get the few first top restaurants in the city to later iterate over them
        try:
            wait_driver.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='RfBGI']/span/a")))
            top_restaurants_in_city_container = driver.find_elements(by=By.XPATH, value=".//div[@class='RfBGI']/span/a")
        except:
            print(f"No top restaurants found for the city {review_city_name}")
            continue

        restaurants_urls = [top_restaurant_in_city.get_attribute('href') for top_restaurant_in_city in top_restaurants_in_city_container]
        print(f"{len(top_restaurants_in_city_container)} top restaurants found on this page for the city {review_city_name}")

        for restaurant_url in restaurants_urls:
            csv_writer_restaurant.writerow([review_city_name, restaurant_url, ""])

    return len(restaurants_urls)

if __name__ == "__main__":
    
    # start time when running the script
    start_time = time.time()

    # get the driver
    driver = TA_utility.get_driver()

    # get the list of selected scrapable cities
    with open("./TripAdvisor/TA_cities.txt", "r") as file:
        cities_to_scrape = [line.rstrip() for line in file]

    nb_restaurants_urls = get_restaurants_urls(cities_to_scrape)
    print(f"{nb_restaurants_urls} restaurants urls found!")

    # finish properly with the driver
    driver.close()
    driver.quit()

    end_time = time.time()
    print("Time elapsed for the run: ", int(end_time - start_time) // 60, " minutes and ", int(end_time - start_time) % 60, " seconds")
