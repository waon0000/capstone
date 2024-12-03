# Knee Arthritis Detection using EfficientNet and ContraD-GAN

## 프로젝트 개요

이 프로젝트는 무릎 관절염 진단을 위해 고해상도 X-ray 이미지를 사용하여 정상, 의심, 경미, 중등도, 심각의 5가지 단계로 분류하는 모델을 개발하는 것을 목표로 합니다. EfficientNetB0을 사용하여 이미지에서 특징을 추출하고, ContraD-GAN을 통해 데이터 품질을 개선함으로써 분류 성능을 향상시키는 것을 중점으로 하고 있습니다.

## 사용 데이터
https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection
kaggle에 있는 무릎 x-ray이미지를 사용하였으며 데이터 세트에는 다양성과 포괄성을 보장하기 위해 다양한 출처에서 수집한 고해상도 무릎 X선 이미지가 포함되어 있습니다.
normal, doubtful, mild, moderate, and severe 

## 주요 코드 설명

이 프로젝트의 주요 코드는 다음과 같은 주요 기능들을 포함하고 있습니다:

- **데이터 수집 및 전처리**: 데이터셋을 로드하고, 전처리를 수행합니다. 이미지를 RGB로 변환하고, 모델에 맞는 크기로 조정한 후, 표준화합니다.
- **EfficientNetB0 기반 모델 학습**: EfficientNetB0을 사전 훈련된 모델로 사용하여 무릎 X-ray 이미지의 특징을 추출합니다.
- **ContraD-GAN을 이용한 데이터 증강**: ContraD-GAN을 사용해 데이터 증강을 수행하고, 이를 통해 모델의 성능을 향상시킵니다.
- **모델 성능 평가**: 학습된 모델을 테스트 데이터셋에 대해 평가하고 혼동 행렬 및 분류 리포트를 생성하여 모델의 성능을 확인합니다.

## 설치 및 실행 방법

프로젝트 코드를 실행하려면, Python 3.8 이상과 필요한 라이브러리를 설치해야 합니다.

1. **의존성 설치**: 아래 명령어를 사용하여 필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
   
2. **데이터 준비**: 코드에서 사용한 무릎 X-ray 데이터셋을 `data/` 디렉토리에 준비합니다.

3. **코드 실행**: 제공된 Python 코드를 실행하여 데이터 전처리, 모델 학습 및 평가를 진행합니다.

## .gitignore 설정

