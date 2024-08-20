import os
import shutil
from PIL import Image
import numpy as np

# 폴더 경로 설정
source_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_label'
destination_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_label_black'

# 목적지 폴더가 없으면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# source_folder 내의 모든 파일에 대해 반복
for filename in os.listdir(source_folder):
    if filename.lower().endswith('.jpg'):
        file_path = os.path.join(source_folder, filename)
        
        # 이미지 열기
        with Image.open(file_path) as img:
            # 이미지를 numpy 배열로 변환
            img_array = np.array(img)
            
            # 이미지가 그레이스케일인지 확인
            is_grayscale = len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1)
            
            # 모든 픽셀이 검은색인지 확인
            if is_grayscale:
                is_black = np.all(img_array == 0)
            else:
                is_black = np.all(img_array == [0, 0, 0])
            
            if is_black:
                # 검은색 이미지를 목적지 폴더로 이동
                shutil.move(file_path, os.path.join(destination_folder, filename))
                print(f"{filename} 이(가) {destination_folder}로 이동되었습니다.")

print("작업이 완료되었습니다.")




# 폴더 경로 설정

# source_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_label'
# destination_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_label_black'
a_folder = 'D:/piston_image/retraining_imgs/land/notfound/a'
b_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_ok'
a_label_folder = 'D:/piston_image/retraining_imgs/land/notfound/a_label'

# b 폴더가 없으면 생성
if not os.path.exists(b_folder):
    os.makedirs(b_folder)

# a_label 폴더 내의 jpg 파일 목록 가져오기
label_files = [f for f in os.listdir(a_label_folder) if f.lower().endswith('.jpg')]

# 각 파일에 대해 처리
for filename in label_files:
    source_path = os.path.join(a_folder, filename)
    dest_path = os.path.join(b_folder, filename)
    
    # a 폴더에 동일한 이름의 파일이 있는지 확인
    if os.path.exists(source_path):
        # 파일을 b 폴더로 이동
        shutil.move(source_path, dest_path)
        print(f"{filename}을(를) a 폴더에서 b 폴더로 이동했습니다.")
    else:
        print(f"{filename}은(는) a 폴더에 존재하지 않습니다.")

print("작업이 완료되었습니다.")