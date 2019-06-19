from selenium import webdriver

#browser = webdriver.Firefox("E:\\Selenium\\geckodriver-v0.24.0-win64\\geckodriver.exe")
#browser = webdriver.Firefox("E:\Selenium\geckodriver.exe")
browser = webdriver.Firefox()

browser.get("https://www.baidu.com")
browser.find_element_by_id("kw").send_keys("selenium")
browser.find_element_by_id("su").click()
browser.quit()
