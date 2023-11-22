# 1. LVDC 마이크로그리드 ESS 충방전 스케쥴링 시스템 

* 이 프로젝트는 직류수용가용 DC 마이크로그리드에서 에너지 저장 시스템(ESS)의 충방전 스케줄링 기능을 구현한 것으로 융합표준연구실에서 개발한 SW입니다.
* 주요 기능은 다음과 같습니다.
* ESS - EMS 간 MQTT 연동
* LSTM, Random Forest, XGBoost 모델을 이용한 전력 사용량 예측 
* 웹 크롤링한 기상청 날씨 정보 기반 태양광 발전량 예측
* 4가지(기본모드, 피크제어, 수요관리, 태양광연계) 충방전 모드별 전력지령치 계산 
* 테스트용 데모 PMS 시뮬레이터 제공 
* PQMS 히스토리 데이터 시각화


## 1.1 Table of Contents
- [Environment & Prerequisites](#environment-and-prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

---

# 2. 설치 방법
## 2.1 Environment and Prerequisites

- 개발언어:  Python
- OS:  Windows 10 / Ubuntu
- Protocol:  MQTT
- DB:  MariaDB


## 2.2 Installation 

- 소스코드를 다운로드 받거나 git 명령어를 통해 소스코드를 다운받습니다. 

```bash
    git clone https://github.com/city256/lvdc.git
```

---

# 3. Usage 

- 다운받은 소스코드의 루트 디렉토리에서 main.py를 실행시킵니다.

```bash
    python3 main.py
```

- PMS 데모 프로그램 실행 방법

```bash
    python3 test/pms_demo.py
```

---

# 4. Authors

* 정상우  jsw256@etri.re.kr
* 안윤영  yyahn@etri.re.kr
* 김성혜  shkim@etri.re.kr



