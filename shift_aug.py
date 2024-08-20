import os
from PIL import Image

# 폴더 경로 지정
folder_path = 'D:/piston_image/retraining_imgs/land/notfound/label/temp'

# 폴더 내 파일 리스트 생성
file_list = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 각 파일 처리
for file_name in file_list:
    # 파일 경로 생성
    file_path = os.path.join(folder_path, file_name)
    
    # 이미지 열기
    image = Image.open(file_path)
    
    # 이미지를 그레이스케일로 변환
    image = image.convert('L')
    
    # 4개의 증강 이미지 생성
    augmented_images = [
        Image.new('L', image.size),  # 새 이미지 생성
        Image.new('L', image.size),  # 새 이미지 생성
        Image.new('L', image.size),  # 새 이미지 생성
        Image.new('L', image.size)   # 새 이미지 생성
    ]
    
    # 증강 이미지 이동
    augmented_images[0].paste(image, (0, -20))  # 위로 5픽셀 이동
    augmented_images[1].paste(image, (20, 0))  # 오른쪽으로 5픽셀 이동
    augmented_images[2].paste(image, (0, 20))  # 아래로 5픽셀 이동
    augmented_images[3].paste(image, (-20, 0)) # 왼쪽으로 5픽셀 이동
    
    # 각 증강 이미지 저장
    for i, aug_image in enumerate(augmented_images):
        # 새 파일 이름 생성
        new_file_name = f'{i}_{file_name}'
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # 이미지 저장
        aug_image.save(new_file_path)
        print(f'{file_name} 파일에서 {new_file_name} 파일이 생성되었습니다.')