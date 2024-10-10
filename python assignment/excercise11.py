def first_occurence(s,c):
    for i in range(len(s)):
        if s[i]==c :
            return i
    return-1



def main():
    s=input("enter a string")
    c=input("enter the word for the occurence")
    print(first_occurence(s,c))
if __name__=="__main__":
     main()


