import re
import os
import numpy as np
import pandas as pd

# 머신러닝 모델
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifierCV,
    Perceptron,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier

# 나이브 베이즈 모델
from sklearn.naive_bayes import BernoulliNB, GaussianNB

# 가우시안 프로세스 모델
from sklearn.gaussian_process import GaussianProcessClassifier

# XGBoost 모델
# from xgboost import XGBClassifier  # 추가된 부분

# 모델 평가 및 메트릭
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# 모델 학습 및 튜닝
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# 데이터 전처리
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 시각화
import matplotlib.pyplot as plt

# 기타 클래스 호출
from src.datamerger import DataMerger
from src.datapreprocessor import DataPreprocessor
from src.modeltrainer import ModelTrainer
from src.modelevaluator import ModelEvaluator
from src.visualization import Visualization
from src.utils import ModelUtilities
from src.submission import PredictionSaver

if __name__ == "__main__":
    # 경로 설정
    train_path = "/data/train.csv"
    test_path = "/data/test.csv"
    model_save_path = "/model/ensemble_model.pkl"
    loaded_model_path1 = "/model/pretrained_model.pkl"
    loaded_model_path2 = "/model/pretrained_model2.pkl"
    submission_save_path = "/data/submission.csv"

    # 2. 데이터 전처리
    data_processor = DataPreprocessor(scaling_method="minmax")
    X, y = ModelTrainer().load_data(train_path, target_column="survived")
    X_train_processed = data_processor.preprocess_data(X)

    # 3. 모델 정의 및 학습
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
    )

    trainer = ModelTrainer(model=rf_model)
    model, X_train, X_val, y_train, y_val = trainer.prepare_and_train_model(
        X_train_processed, y
    )
    trainer.save_model(model_save_path)

    # 4. 저장된 모델 불러오기 및 앙상블
    model_util = ModelUtilities()
    loaded_model1 = model_util.load_model(loaded_model_path1)
    loaded_model2 = model_util.load_model(loaded_model_path2)

    # 앙상블 모델 정의
    ensemble_models = {
        "RandomForest": rf_model,
        "LoadedModel1": loaded_model1,
        "LoadedModel2": loaded_model2,
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "GaussianProcess": GaussianProcessClassifier(),
        "LogisticRegressionCV": LogisticRegressionCV(solver="liblinear", max_iter=1000),
        "RidgeClassifierCV": RidgeClassifierCV(),
        "Perceptron": Perceptron(),
        "BernoulliNB": BernoulliNB(),
        "GaussianNB": GaussianNB(),
        "KNeighbors": KNeighborsClassifier(),
        "SVC": SVC(probability=True),
        "NuSVC": NuSVC(probability=True),
        "LinearSVC": LinearSVC(dual=False),
        "DecisionTree": DecisionTreeClassifier(),
        "ExtraTree": ExtraTreeClassifier(),
    }

    # Voting Classifier 앙상블 모델 학습 (가중치 적용)
    ensemble_weights = [1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ensemble_model = trainer.voting_classifier_ensemble(
        models=ensemble_models,
        X_train=X_train,
        y_train=y_train,
        weights=ensemble_weights,
    )

    # 5. 모델 평가 및 결과 출력
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.evaluate_model(ensemble_model, X_val, y_val)
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")

    cross_val_acc = evaluator.cross_validation_score(X_train, y_train, ensemble_model)
    print(f"Voting Classifier Cross-Validation Accuracy: {cross_val_acc:.4f}")

    # 6. 시각화
    visualizer = Visualization()
    visualizer.plot_roc_auc(ensemble_model, X_val, y_val)

    # 최종 예측 및 저장
    prediction_saver = PredictionSaver(model=ensemble_model)
    prediction_saver.final_prediction_and_save(
        test_path=test_path,
        save_path=submission_save_path,
        feature_set=X_train_processed.columns,
        preprocess_fn=lambda x: data_processor.preprocess_data(
            x, reference_data=X_train_processed
        ),
        merge_fn=lambda test_path, test_augmented_path: test_path.merge_and_save(),
    )
