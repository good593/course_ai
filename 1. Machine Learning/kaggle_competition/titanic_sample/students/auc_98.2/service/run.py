from preprocess import preprocess_dataset
from process import process
from submit import submit

from data import load_data
from model import get_model


def main():
    # 데이터 로드
    train, test, submission = load_data()  # submission 미사용

    # 데이터 전처리
    (
        X_train_selected,
        X_test_selected,  # 미사용
        X_selected,
        test_selected,
        y_train_smote,
        y_smote,
        y_test,  # 미사용
    ) = preprocess_dataset(train, test)

    # 베이스 모델 가져오기
    get_model(X_train_selected, y_train_smote, X_selected, y_smote)

    # 프로세싱
    process(X_selected, y_smote)

    # 제출 파일 생성
    submit(X_selected, y_smote, test_selected)


if __name__ == "__main__":
    main()
