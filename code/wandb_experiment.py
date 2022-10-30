#######
#미완료#
########

import os 
import numpy as np 
import torch
import matplotlib.pyplot as plt 
import seaborn as sns 

"""
Wandb 자동로그인 하기

[1] pip install wandb 를 통해 wandb 설치하기

[2] 온라인의 wandb에서 본인의 계정 홈페이지 => settings => Danger Zone에 있는 API Keys 가져오기(없으면 발급하면 될 것. 근데 초기에 발급할거에요 아마)

[3] vi /etc/bash.bashrc를 통해 bash의 설정파일에 접속(bashrc란, 터미널을 켤 때 자동으로 load되는 기본 설정들.)

[4] /etc/bash.bashrc의 맨 밑줄에 다음 명령어를 추가

export WANDB_API_KEY=$(YOUR_SECRET_API_KEY_HEREE!!!)

저기 달러부분에 API key를 집어넣기. 문자열 형태로 "asd123fasd123fsadf" 이런 느낌으로 집어넣으면 될 것
띄어쓰기 조심!!

[5] Esc -> :wq!를 통해 수정

[6] exec bash를 통해 바로 업데이트

[7] wandb login을 누르면 자동으로 login이 될겁니다!
"""

import wandb
from dataset import MaskLabels, GenderLabels, AgeLabels  #Label 생성용


class ExperimentWandb:
    def __init__(self, project_name = "MaskClasfficiationBoostcamp", run_name = "Wandb_EfficientB4_baseline"):
        self.project_name = project_name 
        self.batch_size = 32
        self.lr = 1e-4 
        self.epochs = 10 
    
    def config(self, config_dict = None):
        wandb.init(
            project = self.project_name,
            config = dict(epochs = self.epochs,
            batch_size = self.batch_size,
            lr = self.lr) if config_dict is None else config_dict,
        )

        wandb.run.name = self.run_name

    def set_project_name(self, new_name):
        self.project_name = new_name 

    def set_run_name(self, run_name):
        self.run_name = run_name 
    
    def set_hyperparams(self, hyopt_configs):
        for config_name, config_value in hyopt_configs.items():
            setattr(self, config_name, config_value)

    def log_category_classifier_accuracy(self, correct_cnt, total_cnt):
        '''
        Epoch당 Age, Mask, Gender 3개에 대한 클래스별 정확도의 Log 기록을 남깁니다.
        :correct_cnt - MultiClass Correct Count List
        :total_cnt - MultiClass Total Count List
        '''

        total_idx = np.arange(18)

        #Multi-encoding 했던걸 반대로 : 6 x mask_index + 3x gender_index + age_index
        mask_pred_idx = total_idx // 6
        gender_pred_idx = total_idx//6
        age_pred_idx = total_idx//6



    def plot_best_model(self, correct_cnt: np.array, total_cnt: np.array) -> None:
        '''
        best 모델에 대해서 각각 Age, Gender, Mask, MultiClass에 대한 정확도를 bar plot으로 출력합니다.
        :correct_cnt - MultiClass Correct Count List
        :total_cnt - MultiClass Total Count List

        bar plot을 이용!
        '''


    def log_miss_label(self, miss_labels: list) -> None:
        '''
        Valid Dataset에서 잘못 라벨링 한 데이터의 이미지와 추론값을 표로 출력합니다.
        :miss_label - 잘못 라벨링 된 데이터의 정보를 담은 리스트, [(img, label, pred)] 형식으로 저장

        wandb table을 이용함
        '''

    def log_train_sample(self, inputs : torch.tensor, labels : torch.tensor) -> None:
        '''
        Train Dataset의 일부 데이터의 이미지와 라벨을 표로 출력합니다.
        :inputs - 이미지 정보를 담은 텐서
        :labels - 라벨 정보를 담은 텐서

        wandb table을 이용함
        '''

        
        







wandb.login()