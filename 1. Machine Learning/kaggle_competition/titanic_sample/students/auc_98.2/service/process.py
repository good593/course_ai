# 앙상블 학습 및 평가 모듈
import pickle

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from utils import reset_seeds

# 베이스 모델 호출
def __load_base_models():
    final_models = {}

    # model_names = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'SVC', 'ExtraTrees', 'AdaBoost', 'HistGradientBoosting', 'GradientBoosting', 'KNeighbors', 'RidgeClassifier', 'MLPClassifier']
    model_names = [
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "SVC",
        "ExtraTrees",
        "AdaBoost",
        "HistGradientBoosting",
        "GradientBoosting",
    ]

    for model_name in model_names:
        with open(f"./model/{model_name}.pkl", "rb") as file:
            final_models[model_name] = pickle.load(file)

    return final_models


def __get_model_scores(final_models, X_selected, y_smote):
    # 모델들에 대해 교차 검증 점수를 계산
    model_scores = {}

    for name, model in final_models.items():
        scores = cross_val_score(
            model, X_selected, y_smote, cv=10, scoring="accuracy", n_jobs=-1
        )
        mean_score = scores.mean()
        model_scores[name] = mean_score
        print(f"{name} CV Accuracy: {mean_score:.8f}")
    return model_scores


def __get_hard_voting(selected_models, X_selected, y_smote, cv):
    hard_voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in selected_models.items()],
        voting="hard",
    )

    # 하드 보팅 앙상블 모델의 교차 검증
    hard_voting_scores = cross_val_score(
        hard_voting_clf, X_selected, y_smote, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Hard Voting CV Accuracy: {hard_voting_scores.mean():.4f}")

    return hard_voting_clf, hard_voting_scores


def __get_soft_voting(selected_models, X_selected, y_smote, cv):
    soft_voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in selected_models.items()],
        voting="soft",
    )

    # 소프트 보팅 앙상블 모델의 교차 검증
    soft_voting_scores = cross_val_score(
        soft_voting_clf, X_selected, y_smote, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Soft Voting CV Accuracy: {soft_voting_scores.mean():.4f}")

    return soft_voting_clf, soft_voting_scores


def __get_stacking(selected_models, X_selected, y_smote, cv):
    stacking_clf = StackingClassifier(
        estimators=[(name, model) for name, model in selected_models.items()],
        final_estimator=LogisticRegression(random_state=42),
    )

    # 스태킹 앙상블 모델의 교차 검증
    stacking_scores = cross_val_score(
        stacking_clf, X_selected, y_smote, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"Stacking CV Accuracy: {stacking_scores.mean():.4f}")

    return stacking_clf, stacking_scores


def __select_best_model(selected_models, X_selected, y_smote, cv):
    hard_voting_clf, hard_voting_scores = __get_hard_voting(
        selected_models, X_selected, y_smote, cv
    )
    soft_voting_clf, soft_voting_scores = __get_soft_voting(
        selected_models, X_selected, y_smote, cv
    )
    stacking_clf, stacking_scores = __get_stacking(
        selected_models, X_selected, y_smote, cv
    )

    # 각각의 모델 성능 평균 계산
    hard_voting_mean = hard_voting_scores.mean()
    soft_voting_mean = soft_voting_scores.mean()
    stacking_mean = stacking_scores.mean()

    # 가장 성능이 좋은 모델 선택
    if max(hard_voting_mean, soft_voting_mean, stacking_mean) == hard_voting_mean:
        best_model = hard_voting_clf
        print("Best Model: Hard Voting")
    elif max(hard_voting_mean, soft_voting_mean, stacking_mean) == soft_voting_mean:
        best_model = soft_voting_clf
        print("Best Model: Soft Voting")
    else:
        best_model = stacking_clf
        print("Best Model: Stacking")

    return best_model


def __save_best_model(best_model):
    # 모델들을 저장할 디렉토리 설정
    model_save_path = "./model/"
    model_name = "best_model"
    # 모델 저장
    file_path = f"{model_save_path}{model_name}.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(best_model, file)
    print(f"{model_name}이 {file_path}에 저장되었습니다.")


@reset_seeds
def process(X_selected, y_smote):
    # 베이스 모델들 불러오기
    final_models = __load_base_models()
    # 각 모델 스코어 계산
    model_scores = __get_model_scores(final_models, X_selected, y_smote)

    # KFold 정의
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # 일정 스코어 이상 모델들 선택
    selected_models = {
        name: model
        for name, model in final_models.items()
        if model_scores[name] >= 0.89
    }
    best_model = __select_best_model(selected_models, X_selected, y_smote, cv)

    __save_best_model(best_model)
