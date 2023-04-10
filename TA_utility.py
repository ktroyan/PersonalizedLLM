from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

import random
import string


def get_driver():
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

    return driver