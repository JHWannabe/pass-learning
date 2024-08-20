import os
from PIL import Image

def split_image(input_folder, output_folder):
    # 입력 및 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 입력 폴더의 모든 파일에 대해 반복
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            
            # 원본 파일명에서 확장자 분리
            name, ext = os.path.splitext(filename)
            
            # 출력 파일 경로 설정
            output_paths = [os.path.join(output_folder, f"{name}_{i+1}{ext}") for i in range(4)]
            
            # 이미 모든 출력 파일이 존재하면 건너뛰기
            if all(os.path.exists(path) for path in output_paths):
                print(f"파일 {filename}의 모든 부분이 이미 존재합니다. 건너뜁니다.")
                continue
            
            # 이미지 열기
            with Image.open(input_path) as img:
                width, height = img.size
                part_width = width // 4
                
                # 이미지를 4등분하여 저장
                for i in range(4):
                    left = i * part_width
                    right = left + part_width
                    part = img.crop((left, 0, right, height))
                    part.save(output_paths[i])
            
            print(f"{filename} 처리 완료")

# 사용 예
input_folder = "Z:/Data/land/whole"
output_folder = "Z:/Data/land/test/GOOD"
split_image(input_folder, output_folder)