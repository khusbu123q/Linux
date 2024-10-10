def anagrams(s1,s2):
    if (len(s1)==len(s2)):
        for i in range (0,len(s1)):
            pr="false"
            for j in range(0,len(s2)):
                pr="true"
                break
            else:
                continue
    return pr

def main():
    s1=input("Enter the first string:")
    s2=input("enter the second string")
    pr=anagrams(s1,s2)
    if pr=="true":
        print("they are anagrams")
    else  :
        print("they are not anagrams")

if __name__=="__main__":
    main()

