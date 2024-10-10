def decimal_binary(decimal):
    base = 2

    result = ""
    q = decimal
    r = q % base
    result = str(r) + result

    q = q // base

    while q > 0:
        r = q % base
        result = str(r) + result

        q = q // base
    return result
def main():
    decimal=int(input("enter a number"))
    print(decimal_binary(decimal))
if __name__ =="__main__":
    main()



