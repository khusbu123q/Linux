def binary_decimal(n):
    b=n[::-1]
    sum=0
    for i in range(0,len(b)):
        num=int(b[i])*(2**i)
        sum=sum+num
    return sum

def main():
    binary_input=str(input("Enter a binary code: "))
    if binary_input.isdigit():
        decimal=binary_decimal(binary_input)
        print(decimal)
    else:
        print("error")



if __name__ == "__main__":
    main()




