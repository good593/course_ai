import pandas as pd
import numpy as np


class PredictionSaver:
    def __init__(self, model=None):
        self.model = model

    def final_prediction_and_save(
        self,
        test_path,
        test_augmented_path,
        save_path,
        feature_set,
        preprocess_fn,
        merge_fn,
    ):
        """
        최종 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고 결과를 저장하는 함수.

        Parameters:
        - test_path: 테스트 데이터 파일 경로
        - test_augmented_path: 테스트 증강 데이터 파일 경로
        - save_path: 예측 결과를 저장할 파일 경로
        - feature_set: 훈련에 사용된 피처 세트
        - preprocess_fn: 데이터 전처리 함수
        - merge_fn: 데이터 병합 함수
        """
        if self.model is None:
            raise ValueError(
                "Model is not provided. Please provide a model before predicting."
            )

        # 테스트 데이터와 증강 데이터를 병합
        merged_test_data = merge_fn(test_path, test_augmented_path)

        # 'survived' 열이 있으면 제거
        if "survived" in merged_test_data.columns:
            merged_test_data = merged_test_data.drop(columns=["survived"])

        # 데이터 전처리 (훈련 데이터와 동일한 피처 집합 사용)
        test_data_processed = preprocess_fn(merged_test_data)

        # 누락된 피처 채우기
        missing_cols = list(set(feature_set) - set(test_data_processed.columns))
        # print("Missing columns that will be added:", missing_cols)

        for col in missing_cols:
            if np.issubdtype(test_data_processed[col].dtype, np.number):
                # 수치형 데이터의 경우, 중앙값으로 채우기
                test_data_processed[col] = test_data_processed[col].median()
            else:
                # 범주형 데이터의 경우, "missing"이라는 새로운 범주로 채우기
                test_data_processed[col] = "missing"

        # 추가된 컬럼 제거 (훈련에 없었던 컬럼들)
        extra_cols = list(set(test_data_processed.columns) - set(feature_set))
        test_data_processed = test_data_processed.drop(columns=extra_cols)

        # 피처 순서를 맞춤
        # print("Final feature order being used for prediction:", feature_set)
        test_data_processed = test_data_processed[feature_set]

        # 예측 수행
        test_predictions = self.model.predict_proba(test_data_processed)  # predict

        # 결과 저장
        submission = pd.DataFrame(
            {
                "passengerid": merged_test_data["passengerid"],
                "survived": test_predictions,
            }
        )

        submission.to_csv(save_path, index=False)
        print(f"예측 결과가 {save_path} 파일로 저장되었습니다.")
