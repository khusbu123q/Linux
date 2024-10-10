def decimal_binary_1s(n):

    count = 0
    while n > 0:
        count += n % 2
        n = n // 2
    return count

def main():
    n=int(input("enter the numbers"))
    print(decimal_binary_1s(n))
if __name__=="__main__":
    main()