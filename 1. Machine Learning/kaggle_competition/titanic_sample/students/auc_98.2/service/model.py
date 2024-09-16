# HPO진행 후 base model 생성 모듈
import pickle

import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils import reset_seeds
from xgboost import XGBClassifier

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    ),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "SVC": SVC(probability=True, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    # 'KNeighbors': KNeighborsClassifier(n_jobs=-1),  # K-Nearest Neighbors 추가
    # 'RidgeClassifier': RidgeClassifier(random_state=42),  # Ridge Classifier 추가
    # 'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000)  # Multi-layer Perceptron 추가
}


# 사전 hpo - optuna
def __pre_hpo(trial, model_name, X, y):
    model = MODELS[model_name]
    # 주의 진짜 매우 아주 오래 걸림
    # 모델에 따라 하이퍼파라미터 범위 설정
    if model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.1, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model.set_params(**params)

    elif model_name in ["RandomForest", "ExtraTrees"]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
        model.set_params(**params)

    elif model_name == "LightGBM":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        }
        model.set_params(**params)

    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int(
                "iterations", 50, 300
            ),  # 더 적은 범위로 줄임
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),  # learning_rate 범위 조정
        }
        model.set_params(**params)

    elif model_name == "LogisticRegression":
        params = {"C": trial.suggest_float("C", 0.001, 10)}
        model.set_params(**params)

    elif model_name == "SVC":
        params = {
            "C": trial.suggest_float("C", 0.1, 10),
            "gamma": trial.suggest_float("gamma", 0.001, 1.0),
        }
        model.set_params(**params)

    elif model_name == "AdaBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        }
        model.set_params(**params)

    elif model_name == "HistGradientBoosting":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
        }
        model.set_params(**params)

    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model.set_params(**params)

    elif model_name == "KNeighbors":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
        }
        model.set_params(**params)

    elif model_name == "RidgeClassifier":
        params = {"alpha": trial.suggest_float("alpha", 0.1, 10.0)}
        model.set_params(**params)

    elif model_name == "MLPClassifier":
        params = {
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01),
        }
        model.set_params(**params)

    pipeline = Pipeline(steps=[("model", model)])

    # 교차 검증을 통한 모델 성능 평가
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    return scores.mean()


# 사전 hpo 실행
def __run_prehpo(X_train_selected, y_train_smote):
    # 하이퍼파라미터 최적화 수행
    best_params = {}
    for model_name in MODELS.keys():
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: __pre_hpo(trial, model_name, X_train_selected, y_train_smote),
            n_trials=50,
            n_jobs=-1,
        )  # n_trials로 조절
        best_params[model_name] = study.best_params
        print(f"Best params for {model_name}: {best_params[model_name]}")


# 메인 hpo - gridsearch
def __run_hpo(best_params, X_selected, y_smote):
    """
    주어진 모델들에 대해 GridSearchCV를 사용하여 최적의 하이퍼파라미터를 찾는 함수.

    Parameters:
    models (dict): 모델 이름과 모델 객체로 구성된 딕셔너리.
    best_params (dict): 각 모델에 대한 Optuna로 찾은 최적 파라미터.
    X_selected (DataFrame): 학습용 피처 데이터.
    y_smote (Series): 학습용 타겟 데이터.

    Returns:
    final_models (dict): 각 모델의 최적 하이퍼파라미터로 학습된 모델.
    """
    final_models = {}

    for model_name in MODELS.keys():
        model = MODELS[model_name]

        # 각 모델에 따른 파라미터 그리드 설정
        if model_name in ["RandomForest", "ExtraTrees"]:
            param_grid = {
                "max_depth": [
                    max(1, best_params[model_name]["max_depth"] - 2),
                    best_params[model_name]["max_depth"],
                    best_params[model_name]["max_depth"] + 2,
                ],
                "min_samples_split": [
                    max(2, best_params[model_name]["min_samples_split"] - 1),
                    best_params[model_name]["min_samples_split"],
                    best_params[model_name]["min_samples_split"] + 1,
                ],
                "min_samples_leaf": [
                    best_params[model_name]["min_samples_leaf"] - 1,
                    best_params[model_name]["min_samples_leaf"],
                    best_params[model_name]["min_samples_leaf"] + 1,
                ],
            }

        elif model_name == "XGBoost":
            param_grid = {
                "subsample": [
                    best_params[model_name]["subsample"] * 0.8,
                    best_params[model_name]["subsample"],
                    best_params[model_name]["subsample"] * 1.2,
                ]
            }

        elif model_name == "LightGBM":
            param_grid = {
                "num_leaves": [
                    max(10, best_params[model_name]["num_leaves"] - 10),
                    best_params[model_name]["num_leaves"],
                    best_params[model_name]["num_leaves"] + 10,
                ]
            }

        elif model_name == "CatBoost":
            param_grid = {
                "learning_rate": [
                    best_params[model_name]["learning_rate"] * 0.8,
                    best_params[model_name]["learning_rate"],
                    best_params[model_name]["learning_rate"] * 1.2,
                ]
            }

        elif model_name == "LogisticRegression":
            param_grid = {
                "C": [
                    best_params[model_name]["C"] * 0.8,
                    best_params[model_name]["C"],
                    best_params[model_name]["C"] * 1.2,
                ]
            }

        elif model_name == "SVC":
            param_grid = {
                "C": [
                    best_params[model_name]["C"] * 0.8,
                    best_params[model_name]["C"],
                    best_params[model_name]["C"] * 1.2,
                ]
            }

        elif model_name == "AdaBoost":
            param_grid = {
                "learning_rate": [
                    best_params[model_name]["learning_rate"] * 0.8,
                    best_params[model_name]["learning_rate"],
                    best_params[model_name]["learning_rate"] * 1.2,
                ]
            }

        elif model_name == "HistGradientBoosting":
            param_grid = {
                "max_depth": [
                    max(1, best_params[model_name]["max_depth"] - 2),
                    best_params[model_name]["max_depth"],
                    best_params[model_name]["max_depth"] + 2,
                ]
            }

        elif model_name == "GradientBoosting":
            param_grid = {
                "max_depth": [
                    max(1, best_params[model_name]["max_depth"] - 2),
                    best_params[model_name]["max_depth"],
                    best_params[model_name]["max_depth"] + 2,
                ]
            }

        elif model_name == "KNeighbors":
            param_grid = {
                "n_neighbors": [
                    max(1, best_params[model_name]["n_neighbors"] - 1),
                    best_params[model_name]["n_neighbors"],
                    best_params[model_name]["n_neighbors"] + 1,
                ]
            }

        elif model_name == "RidgeClassifier":
            param_grid = {
                "alpha": [
                    best_params[model_name]["alpha"] * 0.8,
                    best_params[model_name]["alpha"],
                    best_params[model_name]["alpha"] * 1.2,
                ]
            }

        # GridSearchCV 실행
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_selected, y_smote)
        final_models[model_name] = grid_search.best_estimator_

        # 최적 하이퍼파라미터 및 교차 검증 정확도 출력
        print(
            f"Best params for {model_name} after GridSearchCV: {grid_search.best_params_}"
        )
        print(f"Best CV Accuracy for {model_name}: {grid_search.best_score_:.4f}")

    return final_models


# 베이스 모델 저장
def __save_models(final_models):
    # 모델들을 저장할 디렉토리 설정
    model_save_path = "./model/"

    # 모델 저장
    for model_name, model in final_models.items():
        file_path = f"{model_save_path}{model_name}.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        print(f"Model {model_name} saved to {file_path}")


@reset_seeds
def get_model(X_train_selected, y_train_smote, X_selected, y_smote):
    best_params = __run_prehpo(X_train_selected, y_train_smote)
    final_models = __run_hpo(best_params, X_selected, y_smote)
    __save_models(final_models)
