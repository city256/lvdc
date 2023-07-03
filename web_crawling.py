from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

# 웹 드라이버 경로 설정 (여기서는 Chrome을 사용하였습니다.)
# 웹 드라이버는 각 브라우저의 공식 웹사이트에서 다운로드 받을 수 있습니다.
driver = webdriver.Chrome()

# 웹 드라이버 초기화

# 웹페이지 접속
url = 'http://bd.kma.go.kr/kma2020/fs/energySelect2.do?menuCd=F050702000'
driver.get(url)



# 웹페이지 로딩 대기
driver.implicitly_wait(10)

# 웹페이지의 HTML 가져오기
html = driver.page_source

# BeautifulSoup 객체 생성
soup = BeautifulSoup(html, 'html.parser')

test = soup.find(id='install_cap').get('value')


# 원하는 데이터 찾기 (여기서는 예시로 페이지의 모든 텍스트를 출력하였습니다.)
# 이 부분은 웹페이지의 구조에 따라 적절한 선택자를 사용하여 수정해야 합니다.
print(soup.get_text())
print(test)
# 웹 드라이버 종료
driver.quit()
