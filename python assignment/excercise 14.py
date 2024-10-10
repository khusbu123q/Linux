def trim_leading_whitespace(S):
    l1=len(S)
    print("length before trimming", l1)
    s=S.strip()
    l2=len(s)
    print("length after trimming",l2)
    return s


def main():
    S=input("enter the string to be trimmed")
    print(trim_leading_whitespace(S))


if __name__=="__main__":
    main()
