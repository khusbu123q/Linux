import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv("/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"]).values
    y1 = data["disease_score"].values
    y2 = data["disease_score_fluct"].values
    return X, y1, y2


def Training_test(x, y):
    up = int(x.shape[0] * 0.70)
    return x[:up], x[up:], y[:up], y[up:]


def bias_term(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def hypothesis(x, th):
    h_t_sum = []
    for i in range(x.shape[0]):
        h = 0
        for j, k in zip(th, x[i]):
            h += (j * k)
        h_t_sum.append(h)
    return np.array(h_t_sum)


def cost_function(h, y):
    c_f = []
    for x, y1 in zip(h, y):
        c_f.append((x - y1) ** 2)
    return (sum(c_f) / 2)


def Derivative(x, y, h):
    x_t = [list(no) for no in (zip(*x))]
    sum1 = []
    for i in x_t:
        sum2 = 0
        for j, k, l in zip(h, y, i):
            sum2 += (j - k) * l
        sum1.append(sum2)
    return np.array(sum1)


def parameters(th, alp, dervs):
    th_n = []
    for i, j in zip(th, dervs):
        th_n.append(i - alp * j)
    return np.array(th_n)


def r2_score(x, y, th):
    y_m = np.mean(y, axis=0)
    h = hypothesis(x, th)
    num = sum((i - j) ** 2 for i, j in zip(h, y))
    denom = sum((i - y_m) ** 2 for i in y)
    return 1 - (num / denom)


def gradient_descent(X_Train, X_Test, Y_Train, Y_Test, thetas):

    iterations = 100000
    cost_funcs = []

    for iteration in range(iterations):

        h_t = hypothesis(X_Train, thetas)


        c_f = cost_function(h_t, Y_Train)
        cost_funcs.append(c_f)


        der_cf = Derivative(X_Train, Y_Train, h_t)

        if iteration > 0 and (np.isnan(c_f) or cost_funcs[-1] > 1e10):
            print(f"Divergence detected. Stopping gradient descent at {iteration}.")
            break

        if iteration > 0 and abs(cost_funcs[iteration - 1] - cost_funcs[iteration]) < 1e-4:

            break

        alpha = 0.001
        thetas = parameters(thetas, alpha, der_cf)

    np.array(cost_funcs)


    r2 = r2_score(X_Test, Y_Test, thetas)
    return thetas, cost_funcs, r2


def KFold_scratch(X, y1):
    X = bias_term(X)
    n = len(X)
    m = len(X[0])
    folds = 10
    if n % 10 >= 5:
        width = int(np.ceil(n / folds))
    else:
        width = n // folds
    indices = [0]
    for i in range(10):
        if indices[i] + width < (n - 1):
            indices.append(indices[i] + width)
    indices.append(n)
    Folds_Array_X = []
    Folds_Array_Y = []
    for i in range(10):
        Folds_Array_X.append(X[indices[i]:indices[i + 1], :])
        Folds_Array_Y.append(y1[indices[i]:indices[i + 1]])

    R = []
    STAT = {}
    for i in range(folds):
        print(f"-------- FOLD {i + 1} --------")
        X_Test = Folds_Array_X[i]
        Y_Test = Folds_Array_Y[i]
        X_Train = np.vstack([fold for idx, fold in enumerate(Folds_Array_X) if idx != i])
        Y_Train = np.hstack([fold for idx, fold in enumerate(Folds_Array_Y) if idx != i])
        t, c, r = gradient_descent(X_Train, X_Test, Y_Train, Y_Test, np.zeros(X_Train.shape[1]))
        # statistical parameters to check if splitting is proper
        STAT[f"Fold{i + 1}"] = {"Test Mean": np.mean(X_Test), "Test n": len(X_Test), "Training Mean": np.mean(X_Train),
                                "Training n": len(X_Train), "R^2 score": r}
        R.append(round(r, 3))
        # print("")
    print("Statistical parameters:")
    stat = pd.DataFrame(STAT).T
    stat['R^2 score'] = stat['R^2 score'].apply(lambda x: '{:.10f}'.format(x))
    print(stat)
    print(f"Average R^2: {sum(R) / len(R)}")
    # PLOTTING R2 SCORES VS FOLDS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(R) + 1), R, marker="o", label="R^2 per fold")
    plt.xlabel("Fold")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Score Across Folds")
    plt.legend()
    plt.grid(True)
    plt.show()


def min_max_scaling(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)
def main():
    X,y1,y2=load_data()
    X_normalized = min_max_scaling(X)

    KFold_scratch(X_normalized, y1)


if __name__=="__main__" :
    main()