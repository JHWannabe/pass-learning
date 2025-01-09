import os

# 폴더 경로 설정
folder_path = "D:\\piston_image\\Data\\zumul\\test\\GOOD\\mold_b\\origin"  # 여기에 대상 폴더 경로를 입력하세요.

# 폴더 내 파일명 변경
for filename in os.listdir(folder_path):
    if filename.endswith("_INV.bmp"):
        # 새로운 파일명 생성
        new_filename = filename.replace("_INV", "")
        
        # 경로 결합
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # 파일 이름 변경
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")
