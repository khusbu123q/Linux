def prime_number(n):
    i=n-1
    while i > 1:
        if n%i == 0:
            return True

        i=i-1

def main():
    n=int(input("enter a number"))
    if prime_number(n):
        print("not a prime number")
    else:
        print("prime number")
if __name__ =="__main__":
     main()