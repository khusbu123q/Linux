import random
def insertion_sort(a):
    length=len(a)
    for j in range(2,length):
        key=a[j]
        i=j-1
        while i>-1 and a[i]>key:
            a[i+1]=a[i]
            i=i-1
        a[i+1]=key
    return a
def main():
    random_list = [random.randint(0, 150) for _ in range(0, 10)]
    print(random_list)
    print(insertion_sort(random_list))

if __name__ == '__main__':
    main()
