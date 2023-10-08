import datetime
from pytimekr import pytimekr

def check_date(date_str):
    # 문자열에서 날짜 객체로 변환
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    # 해당 날짜가 공휴일인지 확인
    holidays = pytimekr.holidays(date_obj.year)

    # 주말인지 확인
    if date_obj.weekday() >= 5:  # 토요일
        return 1
    elif date_obj in holidays:
        return 1
    return 0

# 예제
print(datetime.datetime.today().date())
date_str = str(datetime.datetime.today().date())  # 원하는 날짜를 입력
print(check_date(date_str))
