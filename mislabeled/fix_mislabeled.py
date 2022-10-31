import pandas as pd
import os

data = pd.read_csv("./mislabeled.csv") #코드랑 같은 폴더에
label_dict = {}
# print(data)
leading_zero = '000000'
for idx in data.index:
    id = str(data.loc[idx]["id"])
    gender = data.loc[idx]["gender"]
    new_id=id
    if(len(id)<5):
        new_id = leading_zero[:6-len(id)]+id
    label_dict[new_id] = gender

pth = "/opt/ml/input/data/train/"
for dic in os.listdir(pth):
    tmp = dic.split('_')
    if tmp[0] not in label_dict: 
        continue
    
    tmp[1] = label_dict[tmp[0]]
    new_name = '_'.join(tmp)
    os.rename(os.path.join(pth+dic), os.path.join(pth+new_name))
    print(new_name)
    # break
