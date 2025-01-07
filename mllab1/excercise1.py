import numpy as np
def transpose_matrix():


    matrix = np.array([[1, 2, 3], [4, 5, 6]])


    transposed_matrix = matrix.T

    print(transposed_matrix)

    multiplication_result=np.dot(transposed_matrix,matrix)
    print(multiplication_result)


transpose_matrix()

