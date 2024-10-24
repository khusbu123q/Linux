import random
def occurences(list,k):
    frequency={}
    for element in list:
        if element in frequency:
            frequency[element]+=1
        else:
            frequency[element]=1
    result=[]
    for element , count in frequency.items():
        if count >k:
            result.append(element)
    return result

def main():
    list=[random.randint(1,10)for _ in range(50)]
    print(list)
    k=int(input("enter the no that needed for frequency "))
    print(occurences(list,k))

if __name__ == '__main__':
    main()