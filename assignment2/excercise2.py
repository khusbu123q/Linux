import random
def max_min(L):

    max=L[0]
    min=L[0]
    for num in L:
        if num > max:
            max=num
        if num < min:
           min=num
    return(max,min)

def main():
    random_list = [random.randint(0, 150) for _ in range(0, 10)]
    print(random_list)
    print(max_min(random_list))

if __name__ == '__main__':
    main()
