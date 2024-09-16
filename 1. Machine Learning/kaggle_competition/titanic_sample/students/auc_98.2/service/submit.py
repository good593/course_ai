import pickle
import pandas as pd


def load_best_model():
    with open("./model/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    return best_model


def __get_predictions(best_model, X_selected, y_smote, test_selected):
    # 최종 모델 학습 및 테스트 데이터에 예측 수행
    best_model.fit(X_selected, y_smote)
    final_predictions = best_model.predict(test_selected)
    return final_predictions


def __save_submission(final_predictions):
    submission = pd.DataFrame(
        {"passengerid": submission["passengerid"], "survived": final_predictions}
    )
    submission.to_csv("submission.csv", index=False)


def __get_prova_predictions(best_model, X_selected, y_smote, test_selected):
    # 최종 모델 학습 및 테스트 데이터에 예측 수행
    best_model.fit(X_selected, y_smote)
    final_predictions = best_model.predict_proba(test_selected)[:, 1]
    return final_predictions


def __save_prova_submission(final_prova_predictions):
    submission = pd.DataFrame(
        {"passengerid": submission["passengerid"], "survived": final_prova_predictions}
    )
    submission.to_csv("prova_submission.csv", index=False)


# 전체 제출 프로세스
def submit(X_selected, y_smote, test_selected):
    # 제일 좋은 모델 로드
    best_model = load_best_model()

    # 타겟값 0, 1 제출파일
    # final_predictions = __get_predictions(
    #     best_model, X_selected, y_smote, test_selected
    # )
    # save_submission(final_predictions)

    # 확률 값 제출파일
    final_prova_predictions = __get_prova_predictions(
        best_model, X_selected, y_smote, test_selected
    )
    __save_prova_submission(final_prova_predictions)
