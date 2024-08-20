import os
from PIL import Image

# 변환할 이미지가 있는 폴더 경로
input_folder = 'D:/piston_image/retraining_imgs/head/notfound/label/temp'  # 원본 이미지 폴더 경로를 여기에 입력하세요
output_folder = 'D:/piston_image/retraining_imgs/head/notfound/label'  # 변환된 이미지를 저장할 폴더 경로를 여기에 입력하세요

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 변환할 새 크기
new_size = (2400, 2400)

# 폴더 내 모든 파일에 대해 반복
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        # 파일 경로 생성
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 이미지를 열고 크기를 변경한 후 저장
        with Image.open(input_path) as img:
            resized_image = img.resize(new_size, Image.Resampling.LANCZOS)
            resized_image.save(output_path)

        print(f'{filename} 변환 완료: {output_path}')

print('모든 이미지 변환이 완료되었습니다.')