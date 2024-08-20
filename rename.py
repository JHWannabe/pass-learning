# import os

# # 폴더 경로를 지정합니다.
# folder_path1 = 'D:/piston_image/Data/skirt/test/GOOD/skirt_a'
# folder_path2 = 'D:/piston_image/Data/skirt/test/GOOD/skirt_b'

# # 폴더 내의 모든 파일을 가져옵니다.
# files1 = os.listdir(folder_path1)
# files2 = os.listdir(folder_path2)

# # jpg 파일을 필터링하고 이름을 변경합니다.
# for file in files1:
#     if file.lower().endswith('.jpg'):
#         if file.startswith('a'):
#             # 파일 이름이 이미 "G"로 시작하는 경우 건너뜁니다.
#             continue
        
#         original_file_path = os.path.join(folder_path1, file)
#         new_file_name = 'N' + file
#         new_file_path = os.path.join(folder_path1, new_file_name)
#         os.rename(original_file_path, new_file_path)

# for file in files2:
#     if file.lower().endswith('.jpg'):
#         if file.startswith('Nb'):   
#             # 파일 이름이 이미 "G"로 시작하는 경우 건너뜁니다.
#             continue
        
#         original_file_path = os.path.join(folder_path2, file)
#         new_file_name = 'Nb' + file
#         new_file_path = os.path.join(folder_path2, new_file_name)
#         os.rename(original_file_path, new_file_path)

# print("파일 이름 변경이 완료되었습니다.")





import os
from tqdm import tqdm

# a 폴더의 경로를 지정합니다. 
# 현재 작업 디렉토리의 'a' 폴더를 기준으로 하고 있습니다.
#folder_path = '//192.168.10.230/JHChun/result/head/super_0802/5/Overkill/origin'
folder_path = 'D:/piston_image/Data/zumul/test/GOOD/mold_a'

# 폴더 내의 모든 jpg 파일 목록을 가져옵니다.
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

# tqdm을 사용하여 진행 상황을 표시합니다.
for filename in tqdm(jpg_files, desc="Processing files"):
    # 파일 이름에 'N'이 있는지 확인합니다.
    if 'a' in filename:
        # 'N'을 제거한 새 파일 이름을 만듭니다.
        new_filename = filename.replace('a', '')
        
        # 원래 파일의 전체 경로
        old_file = os.path.join(folder_path, filename)
        # 새 파일의 전체 경로
        new_file = os.path.join(folder_path, new_filename)
        
        # 파일 이름을 변경합니다.
        os.rename(old_file, new_file)

    if 'G' in filename:
        # 'N'을 제거한 새 파일 이름을 만듭니다.
        new_filename = filename.replace('G', '')
        
        # 원래 파일의 전체 경로
        old_file = os.path.join(folder_path, filename)
        # 새 파일의 전체 경로
        new_file = os.path.join(folder_path, new_filename)
        
        # 파일 이름을 변경합니다.
        os.rename(old_file, new_file)
        #tqdm.write(f"Renamed: {filename} -> {new_filename}")

print("처리가 완료되었습니다.")

# import os

# # 작업할 폴더 경로
# folder_path = 'D:/piston_image/retraining_imgs/head/notfound/a'

# # 폴더 내의 모든 파일에 대해 반복
# for filename in os.listdir(folder_path):
#     # '_segmap'이 파일 이름에 있는 경우에만 처리
#     if 'N' in filename:
#         # 새 파일 이름 생성 (예: 'a_segmap.jpg' -> 'a.jpg')
#         new_filename = filename.replace('N', '')
        
#         # 전체 파일 경로
#         old_file = os.path.join(folder_path, filename)
#         new_file = os.path.join(folder_path, new_filename)
        
#         # 파일 이름 변경
#         os.rename(old_file, new_file)
#         print(f'파일 이름 변경: {filename} -> {new_filename}')

# print('모든 파일 이름 변경 완료')