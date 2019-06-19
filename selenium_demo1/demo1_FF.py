from selenium import webdriver
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

#driver = webdriver.Chrome("E:\Selenium\chromedriver.exe")
driver = webdriver.Firefox()
driver.get("http://www.baidu.com")
print(driver.title)

driver.find_element_by_id("kw").clear()
driver.find_element_by_id("kw").send_keys("Python")
driver.find_element_by_id("su").click()
time.sleep(5)
driver.quit()