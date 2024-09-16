import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from deap import base, creator, tools, gp, algorithms
import operator
import math
import pickle

# 상수 정의
A = 0.058823499828577
B = 0.841127
C = 0.138462007045746
D = 0.31830988618379069
E = 2.810815
F = 0.63661977236758138
G = 5.428569793701172
H = 3.1415926535897931
I = 0.592158
J = 4.869778
K = 0.063467
L = -0.091481
M = 0.0821533
N = 0.720430016517639
O = 0.230145
P = 9.89287
Q = 785
R = 1.07241
S = 281
T = 734
U = 5.3
V = 67.0
W = 2.484848
X = 8.48635
Y = 63
Z = 12.6275
AA = 0.735354
AB = 727
AC = 2.5
AD = 2.6
AE = 0.3
AF = 3.0
AG = 0.226263
AH = 2.0
AI = 12.4148
AJ = 96
AK = 0.130303
AL = 176
AM = 3.2

# 데이터 정리 함수
def CleanData(data):
    data.columns = data.columns.str.lower()  # 칼럼명을 소문자로 변환
    data.drop(["ticket", "name"], inplace=True, axis=1)
    
    # 성별 데이터 처리
    data["gender"] = data["gender"].map({"male": 1, "female": 0}).fillna(0)

    # Cabin 데이터 처리
    data["cabin"] = data["cabin"].fillna("0").str[0].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}).fillna(0)

    # Embarked 데이터 처리
    data["embarked"] = data["embarked"].map({"C": 1, "Q": 2, "S": 3}).fillna(0)

    # 결측값 채우기
    data.fillna(-1, inplace=True)
    
    return data.astype(float)

# GeneticFunction 정의
def GeneticFunction(data, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, AH, AI, AJ, AK, AL, AM):
    return (
        np.minimum((A + data["gender"] - np.cos(data["pclass"] / AH)) * AH, B) * AH
        + np.maximum(data["sibsp"] - AC, -(np.minimum(data["gender"], np.sin(data["parch"])) * data["pclass"]))
        + AG * (np.minimum(data["gender"], ((data["parch"] / AH) / AH)) * data["age"] - data["cabin"])
        + np.minimum(np.sin(data["parch"] * ((data["fare"] - AA) * AH)) * AH, data["sibsp"] / AH)
        + np.maximum(np.minimum(-np.cos(data["embarked"]), C), np.sin((data["cabin"] - data["fare"]) * AH))
        + -np.minimum((data["age"] * data["parch"] * data["embarked"] + data["parch"]), np.sin(data["pclass"]))
        + np.minimum(data["gender"], np.sin(-(data["fare"] * np.cos(data["fare"] * W))) / AH)
        + np.minimum(O, np.sin(np.minimum((V / AH) * np.sin(data["fare"]), D)))
        + np.sin(np.sin(data["cabin"]) * (np.sin(Z) * np.maximum(data["age"], data["fare"])))
        + np.sin(np.minimum(data["fare"], (data["cabin"] * data["embarked"]) / AH) * -data["fare"])
        + np.minimum(AD * data["sibsp"] * np.sin(AJ * np.sin(data["cabin"])), data["parch"])
        + np.sin(np.sin(np.maximum(np.minimum(data["age"], data["cabin"]), data["fare"] * AK) * data["cabin"]))
        + np.maximum(np.sin(AI * (data["age"] / AH)), np.sin(-AF * data["cabin"]))
        + np.minimum(np.sin(data["fare"] * AH * AH * AH), data["sibsp"]) / AH
    )

# 유전 알고리즘 예측 함수
def GP_deap(evolved_train, n_iter=10):
    outputs = evolved_train["survived"].values.tolist()
    inputs = evolved_train.drop(["survived", "passengerid"], axis=1).values.tolist()

    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    # Primitive set 생성
    pset = gp.PrimitiveSet("MAIN", len(inputs[0]))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.tanh, 1)

    # 결과 정의
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSymbReg(individual):
        func = toolbox.compile(expr=individual)
        return (sum(np.round(1.0 - (1.0 / (1.0 + np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / len(outputs),)

    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, n_iter, stats=stats, halloffame=hof, verbose=True)

    return toolbox.compile(expr=hof[0])

# 데이터 전처리 함수
def MungeData(data):
    data.columns = data.columns.str.lower()  # 칼럼명을 소문자로 변환
    data["relatives"] = data.sibsp + data.parch
    data["fare_per_person"] = data.fare / (data.sibsp + data.parch + 1)

    # 연속형 데이터 스케일링
    numeric_features = ["relatives", "fare_per_person", "fare", "age"]
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    # 범주형 데이터 인코딩
    categorical_features = ["pclass", "embarked", "gender", "cabin"]
    encoded_features = []
    for feature in categorical_features:
        encoder = OneHotEncoder().fit_transform(data[feature].values.reshape(-1, 1)).toarray()
        encoded_df = pd.DataFrame(encoder, columns=[f"{feature}_{i}" for i in range(encoder.shape[1])])
        encoded_df.index = data.index
        encoded_features.append(encoded_df)

    data = pd.concat([data, *encoded_features], axis=1)
    return data

# 학습 및 제출 파일 생성 함수
def main():
    raw_train = pd.read_csv("/data/train.csv")
    raw_test = pd.read_csv("/data/test.csv")

    pass_id_train = raw_train["passengerid"]
    survived_train = raw_train["survived"]
    pass_id_test = raw_test["passengerid"]

    evolved_train = MungeData(CleanData(raw_train))
    evolved_test = MungeData(CleanData(raw_test))

    # 유전 알고리즘 모델 훈련
    GeneticFunctionObject = GP_deap(evolved_train)

    # 훈련 데이터 예측 및 정확도 출력
    train_nparray = evolved_train.drop(["passengerid", "survived"], axis=1).values.tolist()
    trainPredictions = np.array([GeneticFunctionObject(*x) for x in train_nparray])
    print("Train set accuracy:", accuracy_score(survived_train.astype(int), np.round(trainPredictions)))

    # 테스트 데이터 예측
    test_nparray = evolved_test.drop("passengerid", axis=1).values.tolist()
    testPredictions = np.array([GeneticFunctionObject(*x) for x in test_nparray])

    pd_test = pd.DataFrame({"passengerid": pass_id_test.astype(int), "survived": np.round(testPredictions).astype(int)})
    pd_test.to_csv("submission.csv", index=False)

    # 모델 저장
    with open("geneticfunction.pkl", "wb") as file:
        pickle.dump(GeneticFunctionObject, file)

if __name__ == "__main__":
    main()