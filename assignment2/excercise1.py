import random
def sumandaverage(l):
    length=len(l)
    sum=0
    for num in range(0,length):
        sum=sum+l[num]
    average=sum/length
    return average,sum

def main():
    random_list = [random.randint(0, 150) for _ in range(0, 10)]
    print(random_list)
    print(sumandaverage(random_list))

if __name__ == '__main__':
    main()
