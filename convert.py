import torch
import torch.nn as nn

# 모델 파일 경로
model_path = 'D:/JHChun/model_weight/zumul/before/supervised_zumul.pt'

# 모델 로드 및 장치 변경
model = torch.load(model_path, map_location='cuda:0')

# 모델을 다시 저장할 파일 경로
new_model_path = 'D:/JHChun/model_weight/zumul/before/supervised_zumul_traced.pth'

# 입력 텐서 생성 (모델의 예상 입력 크기에 맞춰 조정해야 함)
inputs = torch.randn(1, 3, 704, 1856).to('cuda:0')

# 모델 트레이싱
try:
    traced_model = torch.jit.trace(model, inputs)
    # 트레이싱된 모델 저장
    traced_model.save(new_model_path)
    print(f"모델이 {new_model_path}에 저장되었습니다.")
except Exception as e:
    print(f"모델 트레이싱 및 저장 중 오류 발생: {e}")
