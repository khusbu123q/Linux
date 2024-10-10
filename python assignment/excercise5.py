def power_raise_base(x,n):
    result=n**x
    return result
def main():
    n=int(input("enter the number"))
    x=int(input("enter the power"))
    print(power_raise_base(x,n))

if __name__ =="__main__":
    main()

