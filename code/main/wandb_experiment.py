#https://docs.wandb.ai/ref/python/log

import os 
import numpy as np 
import torch
import matplotlib.pyplot as plt 
import seaborn as sns 
import random

"""
Wandb 자동로그인 하기

[1] pip install wandb 를 통해 wandb 설치하기 + pip install plotly



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
    def __init__(self, project_name = "MaskClasfficiationBoostcamp", run_name = "Wandb_EfficientB4"):
        self.project_name = project_name 
        self.run_name = run_name
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

        wandb.run_name = self.run_name

    def set_project_name(self, new_name):
        self.project_name = new_name 

    def set_run_name(self, run_name):
        self.run_name = run_name 
    
    def set_hyperparams(self, hyopt_configs):
        for config_name, config_value in hyopt_configs.items():
            setattr(self, config_name, config_value)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)


    def check_all_accuracy(self, correct_single_cnt, total_single_cnt):
        '''
        Epoch당 Age, Mask, Gender 3개에 대한 클래스별 정확도의 Log 기록을 남깁니다.
        :correct_cnt - MultiClass Correct Count List  [ 3, 7, 12, ....]
        :total_cnt - MultiClass Total Count List
        '''

        total_idx = np.arange(18)

        #Multi-encoding 했던걸 반대로 : 6 x mask_index + 3x gender_index + age_index
        mask_pred_idx = total_idx // 6
        gender_pred_idx = (total_idx- 6* mask_pred_idx) // 3
        age_pred_idx = (total_idx- 6* mask_pred_idx) % 3

        mask_normal = total_idx[mask_pred_idx == MaskLabels.NORMAL]    # ex) [0,1,2,3,4,5]
        mask_incorrect = total_idx[mask_pred_idx == MaskLabels.INCORRECT]
        mask_mask = total_idx[mask_pred_idx == MaskLabels.MASK]

        gender_male = total_idx[gender_pred_idx == GenderLabels.MALE]
        gender_female = total_idx[gender_pred_idx == GenderLabels.FEMALE]

        age_young = total_idx[age_pred_idx == AgeLabels.YOUNG]    #이를테면, 18개 label중 나이가 어린 것들을 다 가져옴
        age_middle = total_idx[age_pred_idx == AgeLabels.MIDDLE]  
        age_old = total_idx[age_pred_idx == AgeLabels.OLD]

        total_correct_ratio = correct_single_cnt.sum() / total_single_cnt.sum()
        
        mask_log = dict(    
            Acc_mask_normal = correct_single_cnt[mask_normal].sum() / total_single_cnt[mask_normal].sum(),
            Acc_mask_incorrect = correct_single_cnt[mask_incorrect].sum() / total_single_cnt[mask_incorrect].sum(),
            Acc_mask_mask = correct_single_cnt[mask_mask].sum() / total_single_cnt[mask_mask].sum()
        )
        
        gender_log = dict(    
            Acc_gender_male = correct_single_cnt[gender_male].sum() / total_single_cnt[gender_male].sum(),
            Acc_gender_female = correct_single_cnt[gender_female].sum() / total_single_cnt[gender_female].sum(),
        )

        age_log = dict(    
            Acc_age_young = correct_single_cnt[age_young].sum() / total_single_cnt[age_young].sum(),
            Acc_age_middle = correct_single_cnt[age_middle].sum() / total_single_cnt[age_middle].sum(),
            Acc_age_old = correct_single_cnt[age_old].sum() / total_single_cnt[age_old].sum(),
        )

        total_log = dict(Acc_total = total_correct_ratio)

        logs_to_write = {**mask_log, **gender_log, **age_log, **total_log}

        """
        wandb로 지표 추적!
        """
        wandb.log(logs_to_write)

        return mask_log, gender_log, age_log, total_log


    def single_plot_image(self, log_type):
        """
        log에 하고자 하는 클래스만 이용하면 충분!
        """
        fig, ax = plt.subplots(1,1, figsize = (12,12))
        
        #Acc_age_old, Acc_total 이런 느낌으로 2번째 것을 추출하면 충분
        image_type = list(log_type.keys())[0].split("_")[1].upper()
        title = f"{image_type} Classify"

        length = len(log_type)

        original_x_values = list(range(length))
        colorbars = ["skyblue", "violet", "purple", "salmon", "magenta", "green", "blue", "brown", "black"]
        clist = random.sample(colorbars, length)

        ax.set_xticks(original_x_values)

        x_values = list(log_type.keys())
        y_values = list(log_type.values())

        #모든 prediction은 0과 1사이
        ax.set_ylim(0,1)
        ax.set_xticklabels(x_values, fontsize = 12)

        #그래프 그리기
        ax.bar(x_values, y_values, color = clist)

        for x, y in zip(original_x_values, y_values):
            ax.text(x, y+ 0.05, str(round(y,2)), color=colorbars[x], fontweight='bold', ha = "center", fontsize = 15)

        #경계선 제거
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.set_title(title, fontsize = 20)
        plt.savefig(f"{image_type}.png",dpi=300)
        return fig, ax

    def multiple_plot_image(self, log_types):
        """
        log에 하고자 하는 클래스만 이용하면 충분!
        """
        fig, ax = plt.subplots(1,3, figsize = (20,8))
        
        fig.suptitle("Multiple class accuracy", fontsize = 25)
        
        for idx, log_type in enumerate(log_types):
            #Acc_age_old, Acc_total 이런 느낌으로 2번째 것을 추출하면 충분
            image_type = list(log_type.keys())[0].split("_")[1].upper()
            title = f"{image_type} Classify"

            length = len(log_type)

            original_x_values = list(range(length))
            colorbars = ["skyblue", "violet", "purple", "salmon", "magenta", "green", "blue", "brown", "black"]
            clist = random.sample(colorbars, length)

            ax[idx].set_xticks(original_x_values)

            x_values = list(log_type.keys())
            y_values = list(log_type.values())

            #모든 prediction은 0과 1사이
            ax[idx].set_ylim(0,1)
            ax[idx].set_xticklabels(x_values, fontsize = 8)

            #그래프 그리기
            ax[idx].bar(x_values, y_values, color = clist)

            for x, y in zip(original_x_values, y_values):
                ax[idx].text(x, y+ 0.05, str(round(y,2)), color=colorbars[x], fontweight='bold', ha = "center", fontsize = 15)

            #경계선 제거
            ax[idx].spines.right.set_visible(False)
            ax[idx].spines.top.set_visible(False)

            ax[idx].set_title(title, fontsize = 15)
        
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        plt.savefig("Multiple.png",dpi=300)

        return fig, ax

    def plot_best_accuracy(self, correct_cnt: np.array, total_cnt: np.array) -> None:
        """
        All accuracy
        """
        
        plt.rcParams['figure.dpi'] = 150

        mask_data, gender_data, age_data, total_data = self.check_all_accuracy(correct_cnt, total_cnt)

        fig, ax = plt.subplots(1,3, figsize = (20,8))

        fig.suptitle("Multiple class accuracy", fontsize = 25)

        log_types = [mask_data, gender_data, age_data]
    
        fig_3way , _ = self.multiple_plot_image(log_types)
        fig_total, _ = self.single_plot_image(total_data)

        wandb.log({"Fig_multi_class" :  fig_3way})
        wandb.log({"Fig_single" :  fig_total})



    def log_miss_label(self, miss_labels: list) -> None:
        '''
        Valid Dataset에서 잘못 라벨링 한 데이터의 이미지와 추론값을 표로 출력합니다.
        :miss_label - 잘못 라벨링 된 데이터의 정보를 담은 리스트, [(img, label, pred)] 형식으로 저장

        wandb table을 이용함
        '''
        missing_table = wandb.Table(columns = ["image", "label", "pred", "Age", "Gender", "Mask"])

        for image, label, pred in miss_labels:
            image = image.transpose(1,2,0)
            image = wandb.Image(image)

            mask_label , mask_pred = label//6  , pred //6
            gender_label , gender_pred = (label - 6 * mask_label) //3 ,  (pred - 6 * mask_pred) //3 
            age_label, age_pred = (label- 6 * mask_label) % 3   , (pred - 6 * mask_pred) % 3

            mask_content = f"Mask_GT : {mask_label}, Mask_pred : {mask_pred}"
            gender_content = f"gender_GT : {gender_label}, gender_pred : {gender_pred}"
            age_content = f"Age_GT : {age_label}, Age_pred : {age_pred}"

            missing_table.add_data(image, label, pred, age_content, gender_content, mask_content)

        wandb.log({"Miss Table": missing_table}, commit = False)
    
    def finish(self):
        wandb.finish()
  

wandb.login()