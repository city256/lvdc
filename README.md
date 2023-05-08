# ESS 충방전 알고리즘
LVDC 마이크로그리드 ESS연계모드 개발


# 1. 운영모드 #

---
모든 운영모드들은 태양광이 연계됨

모든 운영모드 함수의 모든 리턴값은 Power Reference 값을 반환함

### 1.1 효율모드 
특정 시간에 ess 충전 그외의 시간에는 방전

### 1.2 피크제어
### 1.3 수요관리
수요관리(DSM, Demand Side Management)란? 최소의 비용으로 소비자의 전기에너지 서비스 욕구를 충족시키기 위하여 소비자의 전기사용 패턴을 합리적인 방향으로 유도하기 위한 전력회사의 제반활동이라고 정의하고 있다.
### 1.4 태양광 연계

load_predict()
pv_predict()

peak_mode()
demand_mode()
optimize_mode()
pv_mode()
passive_mode()

Pandas, pymysql, paho-MQTT 활용 