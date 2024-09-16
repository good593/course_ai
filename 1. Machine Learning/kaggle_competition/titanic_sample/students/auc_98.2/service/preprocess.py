# 데이터 전처리 실행 모듈
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from utils import reset_seeds

DICT_INITIAL_NAME = {
    "Mr": "Others",
    "Miss": "Others",
    "Mrs": "Others",
    "Master": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Jonkheer": "Royalty",
    "Countess": "Royalty",
    "Sir": "Royalty",
}

# 테스트할 임계값 목록
SELECT_THRESHOLDS = [0.005, 0.0075, 0.01, 0.015, 0.02]
# 테스트할 임계값
COLLELATION_THRESHOLD = 0.999


# __cleaning_data에서 사용하는 함수
# title을 카테고리로 분류하는 기능
def __add_initial_name_type(initial_name):
    result = "Others"
    try:
        result = DICT_INITIAL_NAME[initial_name]
    except:
        pass
    return result


# 데이터 클리닝
def __cleaning_data(df):
    """
    주어진 데이터프레임을 클리닝하는 함수.
    - 결측치 처리
    - 이상치 처리
    - 불필요한 칼럼 제거
    - 호칭 추출 및 처리

    Parameters:
    df (pd.DataFrame): 원본 데이터프레임

    Returns:
    pd.DataFrame: 클리닝된 데이터프레임
    """

    # 이름에서 호칭(title) 추출
    df["title"] = df["name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df["Initial_name_type"] = df["title"].map(lambda x: __add_initial_name_type(x))

    # cabin: 'unknown'으로 대체
    df["cabin"].fillna("unknown", inplace=True)

    # age: title별로 중간값으로 대체 (transform을 사용하여 인덱스 문제 해결)
    df["age"] = df.groupby("title")["age"].transform(lambda x: x.fillna(x.median()))

    # age: 중간값으로 대체 (transform을 사용하여 인덱스 문제 해결)
    df["age"].fillna(df["age"].median(), inplace=True)

    # fare: 중간값으로 대체
    df["fare"].fillna(df["fare"].median(), inplace=True)

    # embarked: 가장 빈번한 값인 's'로 대체
    embarked_imputer = SimpleImputer(strategy="most_frequent")
    df["embarked"] = embarked_imputer.fit_transform(df[["embarked"]]).ravel()

    # 'fare' 값이 500이 넘는 경우를 300 이하에서 가장 높은 값으로 변환
    max_fare_below_300 = df[df["fare"] <= 300]["fare"].max()
    df.loc[df["fare"] > 500, "fare"] = max_fare_below_300

    return df


# 커스텀 피처 생성
def __create_custom_features(df):
    """
    주어진 데이터프레임에 대해 피처 엔지니어링을 수행하는 함수.
    - family_size, is_alone, fare_per_person, deck, age_bins, pclass_age, pclass_fare, shared_ticket_count, ticket_prefix, has_cabin, embarked_fare_median, is_child 등의 새로운 피처를 생성

    Parameters:
    df (pd.DataFrame): 원본 데이터프레임

    Returns:
    pd.DataFrame: 피처 엔지니어링이 완료된 데이터프레임
    """

    # 1. 가족 크기(family_size) 변수 생성
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_small_family"] = (
        (df["family_size"] >= 2) & (df["family_size"] <= 4)
    ).astype(int)
    df["family_size_category"] = pd.cut(
        df["family_size"],
        bins=[0, 1, 4, df["family_size"].max()],
        labels=["Alone", "Small", "Large"],
    )

    # 2. IsAlone 변수 생성
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # 3. Fare 변수 생성
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["fare_bin"] = pd.qcut(df["fare"], 4, labels=False)

    # 4. Deck 변수 생성
    # df['deck'] = df['cabin'].apply(lambda x: x[0].upper() if pd.notna(x) and x != 'unknown' else 'unknown')

    # 5. Age Binning (나이 구간화)
    df["age_bins"] = pd.cut(
        df["age"],
        bins=[0, 4, 12, 19, 35, 60, np.inf],
        labels=["infant", "child", "teen", "young_adult", "middle_aged", "senior"],
    )
    # df['age_bin_interaction'] = df['age_bins'].astype(str) + '_' + df['pclass'].astype(str)

    # del Shared Ticket Count 변수 생성
    df["shared_ticket_count"] = df.groupby("ticket")["ticket"].transform("count")
    # 2, 3인 경우와 4, 5, 6, 7인 경우를 묶어서 새로운 피처 생성
    df["shared_ticket_group"] = df["shared_ticket_count"].apply(
        lambda x: (
            "group_1" if x == 1 else "group_2_3" if x in [2, 3] else "group_4_plus"
        )
    )
    # del Ticket Prefix (티켓 접두사) 변수 생성
    df["ticket_prefix"] = df["ticket"].apply(
        lambda x: x.split()[0] if not x.isdigit() else "none"
    )

    # 'none'인 경우와 아닌 경우를 구분하는 새로운 피처 생성
    df["has_ticket_prefix"] = df["ticket_prefix"].apply(
        lambda x: "none" if x == "none" else "prefix"
    )

    # 9. Has_Cabin (Cabin 정보 유무) 변수 생성
    df["has_cabin"] = df["cabin"].apply(lambda x: 0 if x == "unknown" else 1)

    # 10. Embarked Fare Median (탑승 항구 별 운임 중간값) 변수 생성
    df["is_southampton_embarked"] = (df["embarked"] == "S").astype(int)
    df["embarked_class_interaction"] = (
        df["embarked"].astype(str) + "_" + df["pclass"].astype(str)
    )

    # 11. 티켓 길이
    df["len_ticket"] = df["ticket"].map(lambda x: len(x.strip().split(" ")))
    df["cabin_is_shared"] = df.groupby("cabin")["cabin"].transform("count") > 1

    # 6. Pclass*Age 및 Pclass*Fare 변수 생성
    df["pclass_age_bins"] = df["pclass"].astype(str) + "_" + df["age_bins"].astype(str)
    df["pclass_fare_bin"] = df["pclass"].astype(str) + "_" + df["fare_bin"].astype(str)
    df["gender_pclass"] = df["gender"].astype(str) + "_" + df["pclass"].astype(str)
    df["gender_age_bins"] = df["gender"].astype(str) + "_" + df["age_bins"].astype(str)
    df["gender_fare_bin"] = df["gender"].astype(str) + "_" + df["fare_bin"].astype(str)
    df["gender_len_ticket"] = (
        df["gender"].astype(str) + "_" + df["fare_bin"].astype(str)
    )
    df["gender_cabin_is_shared"] = (
        df["gender"].astype(str) + "_" + df["fare_bin"].astype(str)
    )
    df["gender_ticket_prefix"] = (
        df["gender"].astype(str) + "_" + df["fare_bin"].astype(str)
    )

    df = df.drop(
        columns=[
            "passengerid",
            "name",
            "title",
            "embarked",
            "fare",
            "age",
            "family_size",
            "age_bins",
            "fare_bin",
            "ticket",
            "cabin",
            "shared_ticket_count",
            "ticket_prefix",
        ]
    )

    return df


# 스케일링 및 변환
def __scale_transform_data(train_df, test_df):
    """
    주어진 데이터프레임을 스케일링하고 범주형 변수를 변환하는 함수.
    - 연속형 변수 스케일링
    - 범주형 변수 인코딩
    - 다항식 피처 생성

    Parameters:
    train_df (pd.DataFrame): 훈련 데이터프레임
    test_df (pd.DataFrame): 테스트 데이터프레임

    Returns:
    pd.DataFrame, pd.DataFrame: 스케일링 및 변환된 훈련 및 테스트 데이터프레임
    """

    # 수치형 변수와 범주형 변수를 자동으로 구별
    continuous_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # 다항식 피처 생성기 설정 (degree=2)
    poly_transformer = PolynomialFeatures(
        degree=2, interaction_only=False, include_bias=False
    )

    # ColumnTransformer를 사용하여 연속형 변수 스케일링, 다항식 변환 및 범주형 변수 인코딩
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), continuous_features),
            ("poly", poly_transformer, continuous_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    # 훈련 데이터에 맞춰 fit하고 동일한 변환을 테스트 데이터에 적용
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    # 칼럼명 처리: 공백을 언더스코어로 치환
    # lightLGM에서 피처 분리 오류 해결
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace(" ", "_") for name in feature_names]

    # 변환된 데이터프레임 반환
    return pd.DataFrame(train_processed, columns=feature_names), pd.DataFrame(
        test_processed, columns=feature_names
    )


# 전체 데이터 처리 및 스케일링 과정
# __cleaning_data, __create_custom_features, __scale_transform_data까지의 과정
def __process_and_scale_data(X_train, X_test, X, test):
    # 데이터 클리닝 함수 적용
    X_train_cleaned = __cleaning_data(X_train)
    X_test_cleaned = __cleaning_data(X_test)
    X_cleaned = __cleaning_data(X)
    test_cleaned = __cleaning_data(test)

    # 피처 엔지니어링 함수 적용
    X_train_engineered = __create_custom_features(X_train_cleaned)
    X_test_engineered = __create_custom_features(X_test_cleaned)
    X_engineered = __create_custom_features(X_cleaned)
    test_engineered = __create_custom_features(test_cleaned)

    # 스케일링 및 트랜스포밍 함수 적용 (다항식 피처 생성 포함)
    X_train_scaled, X_test_scaled = __scale_transform_data(
        X_train_engineered, X_test_engineered
    )
    X_scaled, test_scaled = __scale_transform_data(X_engineered, test_engineered)

    return X_train_scaled, X_test_scaled, X_scaled, test_scaled


# smote : 오버샘플링
# 데이터 불균형을 맞추는 역할
# 불균형 데이터 : 정상 범주의 관측치 수와 이상 범주의 관측치 수가 현저히 차이나는 데이터
# 데이터의 개수가 적은 클래스의 표본을 가져온 뒤 임의의 값을 추가하여 새로운 샘플을 만들어 데이터에 추가
# 언더샘플링보다 오버샘플링이 유리한 경우가 많음
# 학습데이터만 샘플링
def __smote_data(X_train_scaled, y_train, X_scaled, y):
    # 4. SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    return X_train_smote, y_train_smote, X_smote, y_smote


# 임계점 교차 검증
def __evaluate_thresholds(X, y, thresholds):
    """
    여러 임계값을 사용하여 피처 중요도를 기반으로 피처를 선택하고, 교차 검증을 통해 성능을 평가하는 함수.

    Parameters:
    X (pd.DataFrame): 설명 변수 데이터프레임
    y (pd.Series): 타깃 변수
    thresholds (list of float): 테스트할 임계값 목록

    Returns:
    dict: 각 임계값에 대한 교차 검증 성능 결과
    """
    results = {}
    model = RandomForestClassifier(random_state=42)

    for threshold in thresholds:
        X_selected, _ = __select_important_features(X, y, threshold=threshold)
        scores = cross_val_score(model, X_selected, y, cv=10, scoring="accuracy")
        results[threshold] = np.mean(scores)

    return results


# 최고 성능의 threshold
# threshold = 임계값
def __get_best_threshold(X_train_smote, y_train_smote):

    # 각 임계값에 대한 성능 평가
    results = __evaluate_thresholds(X_train_smote, y_train_smote, SELECT_THRESHOLDS)

    # 결과 출력
    for threshold, score in results.items():
        print(f"Threshold: {threshold}, Cross-Validation Accuracy: {score:.8f}")

    # 최적의 임계값 선택
    best_threshold = max(results, key=results.get)
    print(f"Best threshold: {best_threshold}")
    return best_threshold


# 중요 피처 선택
def __select_important_features(X, y, threshold=0.01):
    """
    랜덤 포레스트를 사용하여 피처 중요도를 계산하고, 중요도가 낮은 피처를 제거하는 함수.

    Parameters:
    X (pd.DataFrame): 설명 변수 데이터프레임
    y (pd.Series): 타깃 변수
    threshold (float): 중요도 임계값. 이 값보다 낮은 피처는 제거됩니다.

    Returns:
    pd.DataFrame: 중요도가 높은 피처만 남긴 데이터프레임
    list: 선택된 중요한 피처의 이름 목록
    """
    # CatBoostClassifier
    model = CatBoostClassifier(random_state=42, verbose=0)
    # model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)

    # 피처 중요도 계산
    feature_importances = model.feature_importances_

    feature_importances_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    )
    feature_importances_df = feature_importances_df.sort_values(
        by="Importance", ascending=False
    )

    # 피처 중요도를 시각화
    # fig = px.bar(
    #     feature_importances_df,
    #     x='Importance',
    #     y='Feature',
    #     orientation='h',
    #     title='Feature Importances',
    #     width=800,
    #     height=600
    # )
    # fig.show()

    # 중요도가 임계값보다 높은 피처만 선택
    important_features = feature_importances_df[
        feature_importances_df["Importance"] > threshold
    ]["Feature"].tolist()
    X_selected = X[important_features]

    return X_selected, important_features


# 상관관계가 높은 데이터 삭제
def __remove_similar_data(X_train_selected, X_test_selected, X_selected, test_selected):
    # 상관관계 계산
    correlation_matrix = X_train_selected.corr()

    high_corr_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > COLLELATION_THRESHOLD:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    # 상관관계가 높은 피처 제거
    X_train_selected = X_train_selected.drop(columns=high_corr_features)
    X_test_selected = X_test_selected.drop(columns=high_corr_features)
    X_selected = X_selected.drop(columns=high_corr_features)
    test_selected = test_selected.drop(columns=high_corr_features)
    return X_train_selected, X_test_selected, X_selected, test_selected


@reset_seeds
def preprocess_dataset(train, test):
    # 'survived'를 타깃 변수로 분리
    y = train["survived"]
    X = train.drop(columns=["survived"])

    # Train/Test 데이터셋으로 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 데이터 처리 및 스케일링 실행
    X_train_scaled, X_test_scaled, X_scaled, test_scaled = __process_and_scale_data(
        X_train, X_test, X, test
    )

    # SMOTE 적용
    X_train_smote, y_train_smote, X_smote, y_smote = __smote_data(
        X_train_scaled, y_train, X_scaled, y
    )

    # 피처 선택 1
    best_threshold = __get_best_threshold(X_train_smote, y_train_smote)
    # 타깃 변수 (y_train)를 가지고 피처 중요도 기반 피처 선택 수행
    X_train_selected, important_features = __select_important_features(
        X_train_smote, y_train_smote, best_threshold
    )
    # 테스트 데이터셋에도 동일한 피처 선택 적용
    X_test_selected = X_test_scaled[important_features]
    # 오리지날 테스트 데이터셋에도 동일한 피처 선택 적용
    test_selected = test_scaled[important_features]
    X_selected = X_smote[important_features]

    X_train_selected, X_test_selected, X_selected, test_selected = (
        __remove_similar_data(
            X_train_selected, X_test_selected, X_selected, test_selected
        )
    )
    return (
        X_train_selected,
        X_test_selected,
        X_selected,
        test_selected,
        y_train_smote,
        y_smote,
        y_test,
    )
