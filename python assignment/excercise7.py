def individual_digits(p):
    for i in range(len(p)):
        digit=p[i]
        print(digit)
    return "done"


def main():
    n=input("enter the number:")
    print(individual_digits(n))
if __name__ =="__main__":
    main()
