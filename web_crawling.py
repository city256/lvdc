import time

from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 웹 드라이버 경로 설정 (여기서는 Chrome을 사용하였습니다.)
# 웹 드라이버는 각 브라우저의 공식 웹사이트에서 다운로드 받을 수 있습니다.
driver = webdriver.Chrome()

# 웹 드라이버 초기화

# 웹페이지 접속
url = 'http://bd.kma.go.kr/kma2020/fs/energySelect2.do?menuCd=F050702000'
driver.get(url)

# 웹페이지 로딩 대기
driver.implicitly_wait(3)

# 지역 선택 및 클릭
driver.find_element(By.ID, 'install_cap').clear()
driver.find_element(By.ID, 'install_cap').send_keys(0.25)
driver.find_element(By.ID, 'txtLat').send_keys(34.982)
driver.find_element(By.ID, 'txtLon').send_keys(126.690)
driver.find_element(By.ID, 'search_btn').click()


'''area = driver.find_element(By.XPATH,'//*[@id="info_2900000000"]/span')
area.click()'''
time.sleep(10)


# 웹페이지의 HTML 가져오기
html = driver.page_source

# BeautifulSoup 객체 생성
soup = BeautifulSoup(html, 'html.parser')

'''print(driver.find_element(By.ID,'toEnergy').text)
print(driver.find_element(By.ID,'areaName').text)'''
print(driver.find_element(By.ID,'toEnergy').text)

# 웹 드라이버 종료
driver.quit()
