import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


class ModelUtilities:
    def __init__(self):
        """
        ModelUtilities 클래스 초기화.
        """
        self.models = self.define_models()
        self.model = None  # 현재 모델을 저장할 변수

    def load_model(self, filename):
        """
        모델을 파일에서 로드하는 함수.

        Parameters:
        - filename: 모델이 저장된 파일 경로.

        Returns:
        - 모델 객체.
        """
        self.model = joblib.load(filename)
        return self.model

    def save_model(self, save_path):
        """
        학습된 모델을 저장하는 함수.

        Parameters:
        - save_path: 모델을 저장할 파일 경로.
        """
        if self.model is not None:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save.")

    def define_models(self):
        """
        여러 머신러닝 모델을 정의하는 함수.

        Returns:
        - models: 정의된 모델들을 담은 딕셔너리.
        """
        models = {
            "AdaBoost": AdaBoostClassifier(),
            "ExtraTrees": ExtraTreesClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GaussianProcess": GaussianProcessClassifier(),
            "LogisticRegressionCV": LogisticRegressionCV(
                solver="liblinear", max_iter=1000
            ),
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
        return models
