import os
from PIL import Image

# BMP 파일이 있는 폴더 경로
input_folder = 'D:/piston_image/head/notfound/label'  # BMP 파일이 있는 폴더 경로를 여기에 입력하세요
output_folder = 'D:/piston_image/retraining_imgs/head/notfound/label'  # 변환된 JPG 파일을 저장할 폴더 경로를 여기에 입력하세요

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 파일에 대해 반복
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.bmp'):
        # 파일 경로 생성
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(output_folder, output_filename)
        
        # 이미지를 열고 JPG 형식으로 저장
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # BMP는 알파 채널이 있을 수 있으므로 RGB로 변환
            img.save(output_path, 'JPEG')

        print(f'{filename} 변환 완료: {output_path}')

print('모든 BMP 파일의 JPG 변환이 완료되었습니다.')
