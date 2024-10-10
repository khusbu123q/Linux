def high_frequency(s):
    c=s.count(s[0])
    m=s[0]
    n=1
    for i in range(1,len(s)):
        c1=s.count(s[i])
        if c1>c:
            m=s[i]
            n=c1

    return m,n
def main():
    s=input("enter the string :")
    m,n =high_frequency(s)
    print("max occured letter is:" ,m)
    print("num of occcurences :" ,n)

if __name__=="__main__":
    main()

