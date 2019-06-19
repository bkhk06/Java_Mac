from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Firefox() # Get local session of firefox
#browser.get("https://cn.bing.com/") # Load page
browser.get("http://www.baidu.com")
#assert "Yahoo" in browser.title
print(browser.title)
#elem = browser.find_element_by_name("p") # Find the query box
#elem = browser.find_element_by_id('kw')
#elem.send_keys("seleniumhq" + Keys.RETURN)
#time.sleep(0.2) # Let the page load, will be added to the API

browser.find_element_by_id('kw').send_keys('selenium')
#selenium 元素查找find_element_by_id方法，找到元素后输入信息
browser.find_element_by_id('su').click()
time.sleep(10)
#selenium 元素查找find_element_by_id方法，找到元素后进行点击

'''
try:
    browser.find_element_by_xpath("//a[contains(@href,'https://seleniumhq.org')]")
except NoSuchElementException:
    assert 0, "can't find seleniumhq"
'''
browser.close()
