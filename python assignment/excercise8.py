def first_half(n):
    half=len(n)//2
    print(n[:half])
    return "done"





def main():
    n=str(input("enter the string"))
    print(first_half(n))
if __name__=="__main__":
    main()
