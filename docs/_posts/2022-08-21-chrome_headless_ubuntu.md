---
layout: post
mathjax: false
title:  "Chrome in headless mode with Selenium on Ubuntu 20.04"
date:   2022-08-21
---


This assumes that you already have Selenium installed.

1. *Get the necessary version of Chrome* As of the time of writing this (21.08.2022), the latest version of Chromium with a working driver and support for headless mode is 105.0.5195.37. You can get it in Ubuntu Software by searching for Chromium browser and then selecting the "latest/beta" option in the "Source" dropdown in the top-right corner of the application windows.

2. *Get the Chrome driver for selenium* After installing the required version of Chromium (and possibly uninstalling any previous version to avoid conflicts) you can download the chrome driver from https://chromedriver.chromium.org/downloads. Download the one for Chrome 105.0.5195.* and open the archive. This contains a single executable file, which needs to be on PATH. Find a good spot for it on your PATH and put it there:  `echo $PATH`

3. *Test it*. Make sure it all works by executing the following script (from https://stackoverflow.com/a/53657649/14132412):

```
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)
start_url = "https://google.com"
driver.get(start_url)
print(driver.page_source.encode("utf-8"))
driver.quit()

```