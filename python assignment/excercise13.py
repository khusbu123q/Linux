def replace_charcters(S,c,d):
    result=''
    for char in S :
        if char==c:
            result +=d
        else:
            result +=char
    return result

def main():
    S=input("enter the string")
    c=input("enter the letter to search and replace ")
    d=input("enter the letter to replace")
    print(replace_charcters(S,c,d))

if __name__=="__main__":
    main()


