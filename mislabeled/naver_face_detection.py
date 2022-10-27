import os
import requests
import openpyxl as xl

client_id = "client_id"
client_secret = "client_secret"
url = "https://openapi.naver.com/v1/vision/face" # 얼굴감지
# url = "https://openapi.naver.com/v1/vision/celebrity" # 유명인 얼굴인식
headers = {'X-Naver-Client-Id': client_id, 'X-Naver-Client-Secret': client_secret }

def detect_face(path):
    files = {'image': open(path, 'rb')}
    response = requests.post(url,  files=files, headers=headers)
    rescode = response.status_code
    if(rescode==200):
        return response.json()
        # print (response.text)
    else:
        print("Error Code:" + rescode)
        return None

path = "C:\\Users\\YJ\\Desktop\\project1\\normal\\"
wb = xl.Workbook()
sheet = wb.active
sheet.append(['ID', 'gender', 'age', 'gender', 'acc', 'age_min', 'age_max', 'acc'])

try :
    for i, pic in enumerate(os.listdir(path)):
        if i==1000 : break #애플리케이션 하나당 1000건의 처리만 가능하므로 1000개씩 잘라서 해줘야 함. 애플리케이션 개수 제한은 몇갠지 모르겠음.

        info_ = pic.split('_')
        id_ = info_[0]
        gender_ = info_[1]
        age_ = info_[3]
        list = [id_, gender_, age_]    

        pic_path = path+pic
        res = detect_face(pic_path)
        info = res.get("info")
        if info.get("faceCount")!=1 :
            print("faceCount::::::",info.get("faceCount"),"::::",pic)
            sheet.append(list)
            continue
        
        faces = res.get("faces")
        face = faces[0]
        gender = face.get("gender")
        age = face.get("age")
            
        list.append(gender.get("value"))
        list.append(str(gender.get("confidence")))
        list.extend(age.get("value").split('~'))
        list.append(str(age.get("confidence")))
        sheet.append(list)
        if i%10==0 : 
            print(i)

except : #중간에 에러 생기면 바로 저장.
    wb.save(path+"data2.xlsx")

wb.save(path+"data2.xlsx")

