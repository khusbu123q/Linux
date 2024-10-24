import random
def matrix(a,b):
    rows=len(a)
    column=len(a[0])
    matrix=[[0 for _ in range(column)]for _ in  range(rows)]
    for i in range(rows):
        for j in range(column):
            matrix[i][j]=a[i][j]-b[i][j]
    return matrix

def main():
    rows,column=3,3
    a=[[random.randint(0, 150) for _ in range(column)]for _ in range(rows)]
    b=[[random.randint(0, 150) for _ in range(column)]for _ in range(column)]
    print("matrix a:")
    for row in a :
        print(row)
    print("matrix b")
    for row in b:
        print(row)
    result=matrix(a,b)
    print("result")
    for row in result:
        print(row)


if __name__ == '__main__':
    main()