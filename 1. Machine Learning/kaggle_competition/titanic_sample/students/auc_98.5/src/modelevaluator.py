from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    def __init__(self, cv=5):
        self.cv = cv

    def evaluate_model(self, model, X_val, y_val):
        """
        학습된 모델을 검증 세트에 대해 평가하는 함수.
        """
        # 검증 세트에 대한 예측 수행
        y_pred = model.predict(X_val)

        # 평가 지표 계산
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        precision = precision_score(y_val, y_pred, average="weighted")
        recall = recall_score(y_val, y_pred, average="weighted")

        # ROC-AUC는 이진 분류에서만 사용 가능
        if len(set(y_val)) == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
        else:
            roc_auc = "N/A"  # 다중 클래스 분류에서는 ROC-AUC를 계산하지 않음

        # 혼동 행렬 계산 및 시각화
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_
        )
        disp.plot(cmap="Blues")

        # 평가 결과를 딕셔너리로 반환
        evaluation_results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }

        return evaluation_results

    def compare_models(self, X_train, y_train, models):
        """
        여러 모델을 비교하여 성능을 평가하는 함수.
        """
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)
            results[name] = accuracy
            print(f"{name}: Accuracy = {accuracy:.4f}")

        return results

    def cross_validation_score(self, X, y, model):
        """
        k-폴드 교차 검증을 수행하는 함수.
        """
        scores = cross_val_score(model, X, y, cv=self.cv, scoring="accuracy")
        mean_score = scores.mean()
        print(f"Cross-Validation Accuracy: {mean_score:.4f}")

        return mean_score
