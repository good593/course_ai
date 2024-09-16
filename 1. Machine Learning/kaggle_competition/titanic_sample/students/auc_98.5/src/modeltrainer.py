import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    def __init__(self, model=None, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self, filepath, target_column):
        """
        데이터를 로드하고, 피처와 타겟을 분리하는 함수.
        """
        try:
            data = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {filepath} was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the file.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        return X, y

    def prepare_and_train_model(self, X, y):
        """
        데이터 분할 및 주어진 모델을 사용한 학습을 수행하는 함수.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        if self.model is not None:
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        return self.model, X_train, X_val, y_train, y_val

    def save_model(self, save_path):
        """
        학습된 모델을 저장하는 함수.
        """
        if self.model is not None:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save.")

    def train_and_save_model(self, X, y, save_path):
        """
        모델 학습과 저장을 한 번에 수행하는 함수.
        """
        model, X_train, X_val, y_train, y_val = self.prepare_and_train_model(X, y)
        self.save_model(save_path)
        return model, X_train, X_val, y_train, y_val

    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        """
        하이퍼파라미터 튜닝을 위해 Grid Search를 사용하는 함수.
        """
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters found: {best_params}")
        return self.model, best_params

    def voting_classifier_ensemble(self, models, X_train, y_train, weights=None):
        """
        VotingClassifier 앙상블 모델 학습을 수행하는 함수.

        Parameters:
        - models: 앙상블에 포함될 모델들 (딕셔너리 형태).
        - X_train: 훈련 데이터.
        - y_train: 훈련 레이블.
        - weights: 각 모델에 부여할 가중치 (리스트 형태, 기본값=None).

        Returns:
        - voting_clf: 학습된 VotingClassifier 모델.
        """
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting="soft",
            weights=weights,  # 가중치 적용
        )
        voting_clf.fit(X_train, y_train)
        print("VotingClassifier trained successfully.")
        return voting_clf

    def stacking_classifier_ensemble(self, models, X_train, y_train):
        """
        StackingClassifier 앙상블 모델 학습을 수행하는 함수.
        """
        stacking_clf = StackingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            final_estimator=LogisticRegression(),
            cv=5,
        )
        stacking_clf.fit(X_train, y_train)
        print("StackingClassifier trained successfully.")
        return stacking_clf
