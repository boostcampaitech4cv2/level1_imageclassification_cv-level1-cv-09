<div align=center>

![header](https://capsule-render.vercel.app/api?type=waving&text=마스크%20착용상태%20분류&color=7F7FD5&fontColor=FFFFFF&fontSize=50&height=200)

</div> 

# 💘 CV09조

<div align=center>

|<img src="https://user-images.githubusercontent.com/72690566/200118081-7f8e4279-04ef-4269-abde-80b9ea89e87a.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118119-d21769d2-ff0d-4e15-9e6d-aa863e700f36.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118141-2de150f1-98cb-4cbd-8ce8-419c1ebb0678.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118162-f25ae93e-18c1-462f-8298-c6ff5c95ee79.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118175-ba5859db-5a2f-4457-a8e2-878f8cc1140e.png" width="80">|
|:---:|:---:|:---:|:---:|:---:|
|구상모|배서현|이영진|권규보|오한별|
|T4008|T4095|T4155|T4011|T4128|

</div>

# ⭐DEMO

### 사진을 입력받아, 마스크 착용 여부 / 성별 여부 / 나이 여부를 구분하는 모델

<img src = "https://user-images.githubusercontent.com/81371318/201253185-88be2e3e-65c3-459a-b308-70ef5c424f6e.gif">


# 🌳 Folder Structure
```
.
├── CV9조_발표자료_데이터처리.pdf : 발표자료
├── CV기초대회_CV_팀_리포트(9조).pdf : 랩업리포트
├── EDA : Data에 대한 접근이 담겨있는 곳
│   ├── Image_EDA.pdf
│   ├── correctmask.png
│   ├── data_EDA.ipynb
│   └── incorrect_mask.png
├── README.md
├── code
│   ├── __init__.py : Module import를 위해 있음
│   ├── app.py : streamlit demo 영상을 위해 있음
│   ├── app_utils.py : 
│   ├── config.yaml : App demo를 위해 존재
│   ├── confirm_button_hack.py : App의 authentication을 위한 구현
│   ├── main : SOTA 모델을 저장한 곳
│   │   ├── __init__.py
│   │   ├── best_f1_0.7875.pth
│   │   ├── config.json
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── single_way_training.sh
│   │   ├── three_way_inference.sh
│   │   ├── train.py
│   │   ├── utils.py : cutmix 등의 utils를 구현해놓은 곳
│   │   └── wandb_experiment.py
│   ├── predict.py
│   └── v2
├── demos : Preprocessing 결과 등을 분석한 곳
│   ├── MTCNN_demo.ipynb
│   └── image_example2
├── requirements.txt
└── utils : 대회 전처리에 이용한 폴더들을 저장해놓은 곳
    ├── fix_mislabeled.py
    └── naver_face_detection.py
```


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

|전체|문제 정의, 계획 및 타임라인 수립, 모델 튜닝, 아이디어 제시|
|:----------:|:------:|
|구상모 &nbsp;&nbsp;&nbsp;&nbsp;|Face detection & Data imbalance 전략 수립 및 구현,  Data Noise GUI 코드제작, three-way model 베이스라인 제작, Custom CutMix 구현 , 시각화용 wandb 연동 , 발표 자료 제작|
|권규보|train 전략 수립, validation data 학습에 이용, 비율이 랜덤한 CutMix 구현, f1 loss 도입, 앙상블 전략 수립 및 구현|
|배서현|EDA 정리 및 문서화, Data Augmentation 전략 수립|
|오한별|EDA(Data bias, label distribution), loss experiment, arcface loss experiment, 문서 정리|
|이영진|Face detection 전략 수립 및 구현, Data noise 수정 코드 제작, 앙상블 전략 수립 및 구현|

</div>
</div>

# **🔑 문제 정의**

## 1. Domain understanding

- 이 문제는, 본질적으로 Multi-label classification 문제
- 예측해야 하는 특징이 gender / age / mask로 되어있는데, 이 3개의 feature가 독립적이라면 3개 label을 각각 예측하는 모델을 만들 수 있다고 생각했음
- 편의상, 18개의 label을 각각 예측하는 모델을 **Single-model**이라고 한다.
- 3개를 각각 예측한 뒤 이를 inference 시 합치는 모델을 **Three-way-branch-model**이라고 부른다.

## 2. EDA

- EDA결과, 다음과 같은 문제점을 확인했다.

### Problem 1.  Face detection

- [미션1] 코드에서 주어진 openCV detection으로는, 안경 쓴 사람들과 노인을 탐지하지 못했다.
- 대부분의 사진이 중앙으로 정렬되어있었지만, 혹시 모를 Data shift에 대비하여  얼굴 탐지를 잘하는 모델을 찾아야 한다.

- **Approach 1.** opencv보다 얼굴을 잘 따내는 모델을 가져오고 전처리에 이용하자
- **Approach 2.** 그것들이 안경 쓴 사람들 + 노인들을 제대로 탐지하는지 실험해보자

### Problem 2. Data imbalance

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200120477-7f0d5e41-b008-489e-87ad-c75c4f1dbe1e.png">

</div>

- 일부 label은 아예 없으며, 60대 이상의 데이터가 극히 희박함을 발견하였다.

- **Approach 3.** Train & Valid에 균등하게 분배될 수 있는 방법론을 찾자
- **Approach 4.** Data Augmentation을 이용하여 60대 이상의 데이터를 늘리자

### Problem 3. Data mislabel

- 데이터셋의 저작권법상 올리지는 않겠지만, 명시적으로 잘못된 데이터가 몇 개 있었다.

- **Approach 5.** Mislabel을 하나씩 별도로 수정하는 코드를 만들자
- **Approach 6.** 1st-stage model이 예측한 psuedo-label을 이용하여 다시 2nd training을 시키자

## 3. Data Preprocessing

### **Problem Handling : Data mislabel  ⇒ EDA Approach 5**

- NAVER의 API를 이용하여, NAVER의 API와 예측값이 차이나는 idx를 별도로 뽑았다.
- 그 결과 Gender가 다른 것이 500장 정도, Age가 다른 것이 900장 정도 있었다.
- 이를 5명이서 분담하여, 직접 다시 Annotate했다.
    - 이때 GUI 코드를 직접 짜서 수정했는데
- 이후 metadata의 경로를 변경하는 코드를 다시 짰다.

</div>

# 💪 Prototyping

<div align=center>
  
<img src="https://user-images.githubusercontent.com/72690566/200571192-3b56913c-e4bd-43a2-985f-80c578c6796f.png">

</div>

</div>

# ✍️ Modeling

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200574552-23e0080a-2f1a-4688-a7fd-78a52948d37d.png">

</div>

</div>

# 🐣 Main strategy

## [1] DataSet & DataLoader 

### Split by Profile  
사람별로 구분해 train/val에 같은 사람이 들어가는 것을 막아준다. Train dataset의 정보가 validation dataset으로 흘러가는 것을 막아준다고 생각했다.  
### WeightedRandomSampler  
데이터가 불균형하여, label의 분포를 고려하여 균등하게 샘플링해주는 샘플러를 구현하였다.  
### Age_removal 
모델이 보다 확실한 feature를 학습하게끔, 경계값인 20대 후반, 50대 후반 데이터를 제외하고 학습했다.  
### Intra-class CutMix dataset 
같은 label 간의 cutmix를 통해, 전반적인 데이터의 오버샘플링을 꾀하였다. 같은 label 간의 CutMix가 오버샘플링 기법인 SMOTE와 의미론적으로 유사하며, data augmentation과 유사한 효과를 낸다고 생각하였다.  
	
## [2] Data Augmentation

  Data Augmentation은 모델의 오버피팅 가능성을 줄여야 하는 동시에, Test time 때의 데이터셋까지 고려하여 이루어져야한다고 
생각했다. 본 대회의 경우 Train/Test가 동일한 환경에서 제작되었을 것이라 가정하여, 일부 Soft Augmentation 만 사용했다.  

  거의 모든 데이터가 정중앙에 정렬된 사람 얼굴이었고, 따라서 회전 + 상하반전 등의 변환은 일반화 가능성을 떨어뜨린다고 여겼다. 또한 color 관련 변환을 이용하지 않았는데, 우리는 우리의 모델이 마스크와 사람의 얼굴의 색상 차이를 잡아내어 마스크를 분류한다고 여겼기 때문이다. 따라서 ColorJitter 등의 변환은 마스크 탐지의 정확도를 떨어뜨린다고 간주했다.  

  되려, 모델이 확실하게 얼굴의 표현만 학습할 수 있게 전처리 단에서 Face detection 모델인 MTCNN을 이용하였고 비슷한 얼굴들을 더욱 robust하게 구분할 수 있게끔, 이미지 간에 서로 정보를 섞어주는 cutmix를 적극적으로 이용하였다.  

###	Normalize  
ImageNet에서 사용되었던 RGB 픽셀 값의 평균과 표준편차 값을 이용하였다.  
###	MTCNN  
얼굴과 그 주변부 정보를 남겨, 얼굴의 크기를 일관성있게 유지하고 배경의 노이즈를 제거하고자 했다.  
###	CenterCrop  
만약 MTCNN이 얼굴을 탐지 못했다면, 이를 CenterCrop 시켰다.  
###	CutMix  
같은 Label끼리(Oversampling 목적), 또는 다른 Label끼리(Robust training 목적) 이미지를 세로로 잘라, 2개의 이미지를 붙였다.  
###	RandErase  
다 지우지 못한 배경의 noise를 제거하거나, 얼굴 정보를 일부 삭제해 보다 robust한 학습이 가능하게끔 했다.  
###	Resize  
EfficientB4의 기본 입력 사이즈인 (380, 380)에 맞게 사이즈를 조절했다.  

<div align=center>  

![Footer](https://capsule-render.vercel.app/api?type=waving&color=7F7FD5&fontColor=FFFFFF&height=200&section=footer)

</div>
