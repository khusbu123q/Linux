def count_occurence(str,word):
    a=str.split(" ")
    count=0
    for i in range(0,len(a)):
        if (word==a[i]):
            count=count+1
    return count

def main():
    string=input("enter the string")
    word=input("enter the word to search")
    print(count_occurence(string,word))

if __name__=="__main__":
    main()