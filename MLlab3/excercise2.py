import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data():
    df = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    X = df[["age","BMI","BP","blood_sugar","Gender"]]
    y = df["disease_score"]
    return X,y

def hypothesis(x,theta):
    result=np.dot(x,theta)
    return result

def loss(x,y,theta):
    predictions=hypothesis(x,theta)
    error=predictions-y

    cost=(1/2)*np.sum(error**2)
    return cost

def derivative(x,y,theta):
    predictions=hypothesis(x,theta)
    error=predictions-y
    gradients=1/2*np.dot(x.T,error)
    return gradients

def gradient_descant(x,y,alpha=0.001,iterations=1000):

    theta=np.zeros((x.shape[1]))
    cost_graph=[]
    for i in range(iterations):
        costs = loss(x, y, theta)

        gradients = derivative(x, y, theta)
        theta -= alpha * gradients
        cost_graph.append(costs)

        if i%100==0:
            print(f"Iteration {i}: Cost = {costs:.4f}")

    return theta,cost_graph



if __name__ == "__main__":

    alpha = 0.001
    iterations = 1000
    x=np.array([[1,1],[1,2],[1,3]])
    y=np.array([1,2,3])
    final_theta,cost_graph =gradient_descant(x,y,alpha, iterations)
    print("Final parameters (theta):", final_theta)

    plt.plot(range(len(cost_graph)),cost_graph)
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.title("cost function vs number of iterations")
    plt.show()




