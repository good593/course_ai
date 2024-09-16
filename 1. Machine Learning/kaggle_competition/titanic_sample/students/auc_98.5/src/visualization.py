import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class Visualization:
    def __init__(self):
        pass

    def plot_feature_importance(self, model, X_train, top_n=20):
        """
        모델의 상위 n개의 피처 중요도를 시각화하는 함수.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_train.columns[i] for i in indices]

        # 상위 n개의 피처만 시각화
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importance")
        plt.barh(range(top_n), importances[indices][:top_n], align="center")
        plt.yticks(range(top_n), names[:top_n])
        plt.gca().invert_yaxis()  # 중요도가 높은 피처가 위에 오도록 설정
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

    def plot_roc_auc(self, model, X_val, y_val):
        """
        주어진 모델과 검증 데이터를 사용하여 ROC-AUC 곡선을 그리는 함수.
        """
        if hasattr(model, "predict_proba"):
            # 예측 확률 계산
            y_val_prob = model.predict_proba(X_val)[:, 1]

            # ROC 곡선 계산
            fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)

            # ROC-AUC 점수 계산
            roc_auc = roc_auc_score(y_val, y_val_prob)

            # ROC 곡선 그리기
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="red", linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.show()

            # ROC-AUC 점수 출력
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        else:
            print(
                "This model does not support probability prediction, so ROC-AUC cannot be plotted."
            )
