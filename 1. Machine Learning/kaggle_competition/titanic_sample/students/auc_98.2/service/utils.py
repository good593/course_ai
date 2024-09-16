# 유틸 모듈
import os
import random

import numpy as np
import torch


def reset_seeds(func, seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 파이썬 환경변수 시드 고정
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu 연산 무작위 고정
    torch.cuda.manual_seed(seed)  # gpu 연산 무작위 고정
    torch.backends.cudnn.deterministic = True  # cuda 라이브러리에서 Deterministic(결정론적)으로 예측하기 (예측에 대한 불확실성 제거 )

    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_func
