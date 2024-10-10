def sum_of_squares(n):
    sum=0
    for i in range(1,n+1):
         sum=sum+(i*i)
    return sum
def main():
    n=int(input("enter the value of n"))
    print(sum_of_squares(n))

if __name__=="__main__":
    main()


