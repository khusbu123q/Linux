def alternate_charcters(n):
    for i in range (0,len(n),2):
        print(n[i])
    return "done"





def main():
    n=input("enter a string")
    print(alternate_charcters(n))
if __name__=="__main__":
    main()

