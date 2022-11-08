<div align=center>

![header](https://capsule-render.vercel.app/api?type=waving&text=마스크%20착용상태%20분류&color=7F7FD5&fontColor=FFFFFF&fontSize=50&height=200)

</div> 

# 💘 CV09조

<div align=center>

|<img src="https://user-images.githubusercontent.com/72690566/200118081-7f8e4279-04ef-4269-abde-80b9ea89e87a.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118119-d21769d2-ff0d-4e15-9e6d-aa863e700f36.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118141-2de150f1-98cb-4cbd-8ce8-419c1ebb0678.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118162-f25ae93e-18c1-462f-8298-c6ff5c95ee79.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118175-ba5859db-5a2f-4457-a8e2-878f8cc1140e.png" width="80">|
|:---:|:---:|:---:|:---:|:---:|
|[구상모]|[배서현]|[이영진]|[권규보]|[오한별]|
|T4008|T4095|T4155|T4011|T4128|

</div>

# ❓ 프로젝트 개요

## 1. Task 소개

마스크의 올바른 착용은 COVID-19의 확산 방지에 중요하다.
그러나 올바른 착용을 사람이 한 명씩 한 명씩 감시하는 것은 너무 비용이 큰 작업이다.
본 대회는, 주어진 이미지로부터 사람의 성별 & 나이 구간대 & 마스크 착용 여부를 예측하고자 하는 대회이다.
구체적으로는, 사람의 이미지 데이터와 메타데이터가 주어졌을 때, **총 18개의 클래스를 예측하고자 하는 모델을 만들고자 한다.**

## 2. 작업 환경

- 컴퓨팅 환경 : V100 GPU
- 협업 도구 : Notion, Slack, Wandb, GitHub

## 3. 작업의 순서

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200120015-b52eb581-764f-41b0-80fe-b083d9accd0f.png">

</div>
  
강의자료에 주어진 Workflow를 참고하여, 프로젝트 타임라인을 위와 같이 설정하였다.

# ❇️ 프로젝트 팀 구성 및 역할

<div align=center>

|전체|ㄱㄴㄷ|
|:-----:|:------:|
|구상모|ㄱㄴㄷ|
|권규보|ㄱㄴㄷ|
|배서현|EDA 정리 및 문서화, Data Augmentation 전략 수립|
|오한별|EDA(Data bias, label distribution), loss experiment, arcface loss experiment, 문서 정리|
|이영진|ㄱㄴㄷ|

</div>
</div>

# **🔑 문제 정의**

## 1. Domain understanding

- 이 문제는, 본질적으로 Multi-label classification 문제
- 예측해야 하는 특징이 gender / age / mask로 되어있는데, 이 3개의 feature가 독립적이라면 3개 label을 각각 예측하는 모델을 만들 수 있다고 생각했음
- 편의상, 18개의 label을 각각 예측하는 모델을 **Single-mode**l이라고 한다.
- 3개를 각각 예측한 뒤 이를 inference 시 합치는 모델을 **Three-way-branch-model**이라고 부른다.

## 2. EDA

- EDA결과, 다음과 같은 문제점을 확인했다.

### Problem 1.  Face detection

- [미션1 ] 코드에서 주어진 openCV detection으로는, 안경 쓴 사람들과 노인을 탐지하지 못했다.
- 대부분의 사진이 중앙으로 정렬되어있었지만, 혹시 모를 Data shift에 대비하여  얼굴 탐지를 잘하는 모델을 찾아야 한다.

- [ ]  **Approach 1. opencv보다 얼굴을 잘 따내는 모델을 가져오고 전처리에 이용하자**
- [ ]  Approach 2. 그것들이 안경 쓴 사람들 + 노인들을 제대로 탐지하는지 실험해보자

### Problem 2. Data imbalance

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200120477-7f0d5e41-b008-489e-87ad-c75c4f1dbe1e.png">

</div>

- **일부 label은 아예 없으며, 60대 이상의 데이터가 극히 희박함을 발견하였다.**

- [ ]  **Approach 3. Train & Valid에 균등하게 분배될 수 있는 방법론을 찾자**
- [ ]  **Approach 4. Data Augmentation을 이용하여 60대 이상의 데이터를 늘리자**

### Problem 3. Data mislabel

- 데이터셋의 저작권법상 올리지는 않겠지만, 명시적으로 잘못된 데이터가 몇 개 있었다.

- [ ]  **Approach 5. Mislabel을 하나씩 별도로 수정하는 코드를 만들자**
- [ ]  Approach 6. 1st-stage model이 예측한 psuedo-label을 이용하여 다시 2nd training을 시키자

## 3. Data Preprocessing

### **Problem Handling : Data mislabel  ⇒ EDA Approach 5**

- NAVER의 API를 이용하여, NAVER의 API와 예측값이 차이나는 idx를 별도로 뽑았다.
- 그 결과 Gender가 다른 것이 500장 정도, Age가 다른 것이 900장 정도 있었다.
- 이를 5명이서 분담하여, 직접 다시 Annotate했다.
    - 이때 GUI 코드를 직접 짜서 수정했는데
    - **지금 와서 생각해보면, 더 좋은 툴이 있었을 것 같다.**
- 이후 metadata의 경로를 변경하는 코드를 다시 짰다.

</div>

# 💪 Prototyping

<div align=center>
  
<img src="https://user-images.githubusercontent.com/72690566/200120567-ad0038bf-902f-46a7-ba79-aea026542b9d.png">

</div>

</div>

# ✍️ Modeling

-실험 테이블 넣는 것보다 요약본을 넣는 게 나을 것 같다-

</div>

# 🐣 Main strategy

-실험 테이블을 정리할 필요가 있어보인다-



<div align=center>  

![Footer](https://capsule-render.vercel.app/api?type=waving&color=7F7FD5&fontColor=FFFFFF&height=200&section=footer)

</div>
