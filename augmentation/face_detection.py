import os 
from PIL import Image
from google.cloud import vision
import io

############# 사용시 아래 파일 경로 수정 필요
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\YJ\\Desktop\\project1\\api-key.json" #구글 api key 발급. https://cloud.google.com/docs/authentication/api-keys?hl=ko 참고

def detect_faces(path):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    # print('Faces:')
    # print(faces)
    for face in faces:
        #아래는 표정감지. 필요없으니 주석처리
        # print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        # print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        # print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])
        # print('face bounds: {}'.format(','.join(vertices)))

        v = face.bounding_poly.vertices
        return (v[0].x, v[0].y, v[2].x, v[2].y)
        

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return None

path = "C:\\Users\\YJ\\Desktop\\project1\\images\\" #이미지 파일이 있는 폴더 경로
for i, pic in enumerate(os.listdir(path)):      
    pic_path = path+pic

    img = Image.open(pic_path)
    target_path = pic_path.replace('images', 'only_face') #잘라낸 얼굴은 'only_face'폴더로
    
    v = detect_faces(pic_path)
    if(v == None): #얼굴 탐지 못한 경우 'detect_error' 폴더로
        print("Crop Error::::::",pic) 
        img.save(pic_path.replace('images', 'detect_error'))
    else:
        cropped = img.crop(v)
        cropped.save(target_path)

    if i%10 == 0: #잘 돌아가는지 확인용
        print(i)
