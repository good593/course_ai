import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataPreprocessor:
    def __init__(self, feature_set=None, scaling_method=None):
        """
        DataPreprocessor 클래스를 초기화하는 메서드.

        Parameters:
        - feature_set: 피처 세트를 일치시키기 위한 선택적 리스트 (기본값: None)
        - scaling_method: 'standard' 또는 'minmax' 중 하나를 선택해 스케일링 (기본값: None)
        """
        self.feature_set = feature_set
        self.scaling_method = scaling_method
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def fill_missing_with_distribution(self, data, column):
        """
        주어진 컬럼의 분포에 따라 결측치를 채우는 메서드.

        Parameters:
        - data: 데이터프레임
        - column: 결측치를 채울 열 이름

        Returns:
        - data[column]: 결측치가 채워진 열
        """
        missing_mask = data[column].isnull()
        if data[column].dropna().empty:
            data.loc[missing_mask, column] = (
                "missing" if data[column].dtype == object else 0
            )
        else:
            if np.issubdtype(data[column].dtype, np.number):
                data.loc[missing_mask, column] = np.random.choice(
                    data[column].dropna(), size=missing_mask.sum(), replace=True
                )
            else:
                probs = data[column].value_counts(normalize=True)
                data.loc[missing_mask, column] = np.random.choice(
                    probs.index, size=missing_mask.sum(), replace=True, p=probs.values
                )
        return data[column]

    def detect_and_replace_outliers(self, data, column, threshold=3.0):
        """
        Z-score 방법을 사용하여 이상치를 탐지하고, 이상치를 이상치가 아닌 데이터 중
        가장 크거나 작은 값으로 대체하는 메서드.

        Parameters:
        - data: 데이터프레임
        - column: 이상치를 탐지할 열 이름
        - threshold: Z-score 임계값 (기본값: 3.0)

        Returns:
        - data[column]: 이상치가 처리된 열
        """
        mean_value = data[column].mean()
        std_dev = data[column].std()
        z_scores = (data[column] - mean_value) / std_dev
        non_outliers = data.loc[np.abs(z_scores) <= threshold, column]
        max_non_outlier = non_outliers.max()
        min_non_outlier = non_outliers.min()
        data[column] = np.where(z_scores > threshold, max_non_outlier, data[column])
        data[column] = np.where(z_scores < -threshold, min_non_outlier, data[column])
        return data[column]

    def integrate_features(self, X):
        """특정 피처들을 통합하는 함수."""
        X["Initial_name"] = X["name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        X["ticket_length"] = X["ticket"].apply(len)
        X.drop(
            columns=[
                "passengerid",
                "name",
                "ticket",
            ],
            errors="ignore",
            inplace=True,
        )
        return X

    def handle_missing_data(self, X):
        """결측치 비율이 높은 피처를 처리하고, 나머지 결측치를 채우는 함수."""
        high_missing_cols = ["cabin"]
        X[high_missing_cols] = X[high_missing_cols].fillna("missing")
        X["initial_cabin"] = X["cabin"].map(lambda cabin: self.add_initial_cabin(cabin))
        X.drop(["cabin"], axis=1, inplace=True)

        for column in X.columns:
            if X[column].isnull().sum() > 0:
                X[column] = self.fill_missing_with_distribution(X, column)
        return X

    def add_initial_cabin(self, cabin, initial_cabin_unique=None):
        """cabin 값의 첫 글자를 추출하여 initial_cabin으로 변환."""
        result = "unknown"
        try:
            cabin = cabin.strip()
            if initial_cabin_unique is None or cabin[:1] in initial_cabin_unique:
                result = cabin[:1]
        except:
            pass
        return result

    def handle_outliers(self, X):
        """수치형 피처의 이상치를 탐지하고 처리하는 함수."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for column in numeric_cols:
            X[column] = self.detect_and_replace_outliers(X, column)
        return X

    def feature_engineering(self, X):
        """피처 엔지니어링을 수행하는 함수."""
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])

        age_bins = [0, 12, 18, 35, 60, np.inf]
        age_labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
        X_categorical["age_group"] = pd.cut(
            X["age"], bins=age_bins, labels=age_labels
        ).astype(str)

        if "gender" in X_categorical.columns:
            X_categorical["gender_encoded"] = X_categorical["gender"]

        if "sibsp" in X_numeric.columns and "parch" in X_numeric.columns:
            X_numeric["family_size"] = X_numeric["sibsp"] + X_numeric["parch"] + 1

        if "fare" in X_numeric.columns:
            X_numeric["fare"] = X_numeric["fare"] * np.random.uniform(
                0.9, 1.1, X_numeric.shape[0]
            )

        return X_numeric, X_categorical

    def scale_features(self, X_processed, numeric_cols):
        """수치형 피처를 스케일링하는 함수."""
        if self.scaling_method:
            scaler = (
                StandardScaler()
                if self.scaling_method == "standard"
                else MinMaxScaler()
            )
            X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        return X_processed

    def align_features(self, X_processed):
        """feature_set에 따라 피처를 정렬하고 부족한 피처를 채우는 함수."""
        if self.feature_set is not None:
            missing_columns = [
                col for col in self.feature_set if col not in X_processed.columns
            ]

            if missing_columns:
                for col in missing_columns:
                    X_processed[col] = 0

            extra_columns = [
                col for col in X_processed.columns if col not in self.feature_set
            ]
            if extra_columns:
                X_processed = X_processed.drop(columns=extra_columns)

            X_processed = X_processed[self.feature_set]
        return X_processed

    def align_features_with_copy(self, train_data, test_data):
        """
        train_data의 피처 집합에 따라 test_data의 피처를 정렬하고 부족한 피처를 채우는 함수.
        train_data에 있는 피처들 중 test_data에 없는 피처를 test_data에 추가하고, 0으로 채운다.
        """
        test_data = test_data.copy()

        missing_columns_in_test = [
            col for col in train_data.columns if col not in test_data.columns
        ]
        if missing_columns_in_test:
            for col in missing_columns_in_test:
                test_data[col] = 0

        missing_columns_in_train = [
            col for col in test_data.columns if col not in train_data.columns
        ]
        if missing_columns_in_train:
            for col in missing_columns_in_train:
                train_data[col] = 0

        test_data = test_data[train_data.columns]

        return test_data

    def encode_and_combine_features(self, X_numeric, X_categorical, fit=True):
        """범주형 피처를 원-핫 인코딩하고 수치형 피처와 결합하는 함수."""
        if fit:
            X_categorical_encoded = self.encoder.fit_transform(X_categorical)
            self.feature_set = self.encoder.get_feature_names_out()
        else:
            X_categorical = X_categorical.reindex(
                columns=self.encoder.feature_names_in_, fill_value=0
            )
            X_categorical_encoded = self.encoder.transform(X_categorical)

        X_categorical_encoded_df = pd.DataFrame(
            X_categorical_encoded,
            columns=self.encoder.get_feature_names_out(),
            index=X_categorical.index,
        )

        X_processed = pd.concat([X_numeric, X_categorical_encoded_df], axis=1)
        return X_processed

    def preprocess_data(self, X, reference_data=None):
        """전처리 전체 과정을 실행하는 함수."""
        X = self.integrate_features(X)
        X = self.handle_missing_data(X)
        X = self.handle_outliers(X)

        X_numeric, X_categorical = self.feature_engineering(X)

        if reference_data is not None:
            X_processed = self.encode_and_combine_features(
                X_numeric, X_categorical, fit=False
            )
        else:
            X_processed = self.encode_and_combine_features(
                X_numeric, X_categorical, fit=True
            )

        numeric_cols = X_numeric.columns
        X_processed = self.scale_features(X_processed, numeric_cols)

        if reference_data is not None:
            X_processed = self.align_features_with_copy(reference_data, X_processed)
        else:
            X_processed = self.align_features(X_processed)

        return X_processed
