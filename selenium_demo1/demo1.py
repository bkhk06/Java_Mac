from selenium import webdriver
import time

driver = webdriver.Chrome("E:\Selenium\chromedriver.exe")
driver.get("http://www.baidu.com")

driver.find_element_by_id("kw").clear()
driver.find_element_by_id("kw").send_keys("Python")
driver.find_element_by_id("su").click()
time.sleep(5)
driver.quit()