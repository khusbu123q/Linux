import random
def even_no(L):
    num=[i for i in range(0,len(L))if i %2==0]
    return num

def main():
    random_list = [random.randint(0, 150) for _ in range(0, 10)]
    print(random_list)
    print(even_no(random_list))
if __name__ == "__main__":
    main()
