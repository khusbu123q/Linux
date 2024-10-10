def concatenate_strings(string1,string2):
    result=string1+string2
    return result


def main():
    string1=input("enter the first string ")
    string2=input("enter the second string")
    print(concatenate_strings(string1,string2))

if __name__=="__main__":
    main()