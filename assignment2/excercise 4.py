import random
def duplicate(L):
    duplicatelist=[]
    uniquelist=[]
    for i in L :
        if i  not in duplicatelist:
            duplicatelist.append(i)
        if i not in uniquelist:
            uniquelist.append(i)
    return duplicatelist,uniquelist

def main():
    random_list = [random.randint(0, 150) for _ in range(0, 30)]
    print(random_list)
    print(duplicate(random_list))

if __name__ == '__main__':
    main()