import pandas as pd
import numpy as np


def normal_equation(X,y):
    X=np.c_[np.ones((X.shape[0])),X]
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def hypothesis(X,theta):
    return np.dot(X,theta)

def cost(X,y,theta):

    c=1/(2*len(X))
    predic=hypothesis(X,theta)-y.reshape(-1)
    temp_1=predic.transpose()
    result=np.dot(predic,temp_1)

    return c*result




def train_test_split(X, y, test_size=0.30, random_state=999):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]





def r2score(y_true, y_pred):
    tot = np.sum((y_true - np.mean(y_true))**2)
    res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (res / tot)
    return r2



def main():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
    y = df["disease_score"].values

    y=y*100
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=999)
    theta=normal_equation(X_train,y_train)
    print(theta)
    print("\nTRAINING")
    print(f"N = {len(X)}")
    y_pred = hypothesis(np.c_[np.ones((X_test.shape[0])), X_test], theta)
    r2 = r2score(y_test, y_pred)
    print(f"R2 of normal equation score is {r2:.2f} (close to 1 is good)")


if __name__=="__main__":
    main()
