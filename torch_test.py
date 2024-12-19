import torch
# MPS 지원 여부 확인
print(f"MPS 장치 지원 여부: {torch.backends.mps.is_built()}")
print(f"MPS 장치 사용 가능 여부: {torch.backends.mps.is_available()}")

# MPS 장치 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# MPS 장치에서 텐서 생성 및 연산
x = torch.ones(5, device=device)
y = x * 2
print(y)

