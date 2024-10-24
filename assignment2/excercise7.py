import random
def remove_all_occurences(list,element):
    result=[item for item in list if item!=element]
    return result

def main():
    list = [random.randint(1, 10) for _ in range(50)]
    print(list)
    k = int(input("enter the no that need to be removed "))
    print(remove_all_occurences(list,k))

if __name__ == '__main__':
    main()