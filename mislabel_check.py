import pandas as pd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle

df = pd.read_csv("data.csv")
df = df.dropna()

usecols = ["ID", "gender", "age", "gender_diff", "age_diff"]
df = df[usecols]
df["age"] = df["age"].astype(int).astype(str)
df["path"] = "input/data/train/images/" + df["ID"] + "_" + df["gender"] + "_Asian_"+ df["age"]

gender_to_change = []
age_to_change = []

change_gender = df[df["gender_diff"] == "false"]
change_age = df[df["age_diff"] == "false"]

def image_gender_check(start_idx, img_id, gender, path):
    pic = os.listdir(path)[0]
    fullpath = os.path.join(path, pic)
    img = Image.open(fullpath)
    plt.imshow(img)
    plt.show()
    new_gender = "female" if gender == "male" else "male"
    
    while True:
        enter_key = input(f"Original Label: {gender}. ID:{img_id}. You think it's correct? [y -yes/ n-no / p-pause / q- quit]")
        if enter_key.lower() == "y":
            print("Mark as correct")
            return "y"
        elif enter_key.lower() == "n":
            print(f"Mark as incorrect : {gender} -> {new_gender} ")
            return "n"

        elif enter_key.lower() == "p":
            print("Mark as stop. Confusing?")
            return "p"
        
        elif enter_key.lower() == "q":
            print("You choose to stop. Temporary")
            return "q"
        else:
            print("Wrong input. [y|n|p|q] is valid. Press again")
    
def check_image_gender(dataframe,  start_idx=0, end_idx=1000):
    pause_lists = []
    false_lists = []
    
    try:
        for idx, row in dataframe.iterrows():
            if idx>end_idx  : continue        
                
            img_id = row["ID"]
            gender = row["gender"]
            age = row["age"]
            path = row["path"]

            new_gender = "male" if gender == "female" else "female"
            check_key = image_gender_check(idx, img_id, gender, path)
            if check_key == "n":
                false_lists.append((img_id, new_gender))    

            elif check_key == "p":
                pause_lists.append((img_id, new_gender))

            elif check_key == "q":
                print("STOP AT idx:", idx)
                print("Record the result somewhere else, if you are not done")
                break
                
            else:
                print("NOTHING SPECIAL")
    
            clear_output(wait=True)
        
    except KeyboardInterrupt:
        print("Interrupt catch")
        return pause_lists, false_lists, idx
        
    return pause_lists, false_lists, idx


def image_age_check(start_idx, img_id, age, path):
    pic = os.listdir(path)[0]
    fullpath = os.path.join(path, pic)
    img = Image.open(fullpath)
    plt.imshow(img)
    plt.show()
    
    while True:
        enter_key = input(f"Original Label: {int(age)//30},{30 * (int(age)//30)} ~ {30 *(int(age)//30+1)} , with age {age}.ID:{img_id}. You think it's correct? [y -yes/ n-no / p-pause / q- quit]")
        if enter_key.lower() == "y":
            print("Correct!")
            return "y", None
        elif enter_key.lower() == "n":
            print("Wrong label. save your predictions..")
            
            while True:
                new_label = input("Your new label ?? Choose from 0,1,2")
                if new_label not in ["0", "1", "2"] :
                    print("다시 입력하세요")
                    continue
                new_label = int(new_label)
                break
                
            return "n", new_label

        elif enter_key.lower() == "p":
            print("You choose to pause. Discuss with others.")
            return "p", None
        
        elif enter_key.lower() == "q":
            print("You choose to stop. Temporary")
            print("Record the result somewhere else, if you are not done")
            return "q", None
        else:
            print("Wrong input. [y|n|p|q] is valid. Press again")
    

def check_image_age(dataframe, start_idx=0, end_idx=1000):
    try:
        pause_lists = []
        false_lists = []
        for idx, row in dataframe.iterrows():
            if idx< start_idx or idx>end_idx : continue        
            img_id = row["ID"]
            gender = row["gender"]
            age = row["age"]
            path = row["path"]

            new_gender = "male" if gender == "female" else "female"
            check_key, new_age = image_age_check(idx, img_id, age, path)
            if check_key == "n":
                false_lists.append((img_id, new_age))    

            elif check_key == "p":
                pause_lists.append((img_id, age))    

            elif check_key == "q":
                print("STOP AT idx:", idx)
                print("Record the result somewhere else, if you are not done")
                break        
                
            else:
                print("NOTHING SPECIAL")
    
            clear_output(wait=True)
            
    except KeyboardInterrupt:
        print("Interrupt catch")
        return pause_lists, false_lists, idx
            
    return pause_lists, false_lists, idx




#상모
GENDER_START , GENDER_END = 0, 100
AGE_START, AGE_END = 0, 190


# #규보
# GENDER_START , GENDER_END = 101, 200
# AGE_START, AGE_END = 191, 380


# #서현
# GENDER_START , GENDER_END = 201, 300
# AGE_START, AGE_END = 381, 570



# #한별
# GENDER_START , GENDER_END = 301, 400
# AGE_START, AGE_END = 571, 760


# #영진
# GENDER_START , GENDER_END = 401, 500
# AGE_START, AGE_END = 761, 950

### Gender

pause_gender_lists, false_gender_lists, stop_gender_idx =  check_image_gender(change_gender,  start_idx = GENDER_START, end_idx = GENDER_END)

pause_gender_list2 , false_gender_list2, stop_gender_idx2 = pause_gender_lists, false_gender_lists, stop_gender_idx

### Age

pause_age_lists, false_age_lists, stop_age_idx = check_image_age(change_age,  start_idx = AGE_START, end_idx = AGE_END)



result_dict = dict(
    pause_gender = pause_gender_lists,
    false_gender = false_gender_lists,
    pause_age = pause_age_lists,
    false_age = false_age_lists,
    stop_gender = stop_gender_idx,
    stop_age = stop_age_idx
)

with open(file='mislabel.pickle', mode='wb') as f:
    pickle.dump(result_dict, f)