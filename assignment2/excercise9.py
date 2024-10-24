def dictionary(dict):
    for key, value in dict.items() :
        print(f"key:{key},value:{value}")

def main():
    dict={"name": "soma", "age": 23, "city": "Bhopal"}
    dictionary(dict)

if __name__ == '__main__':
    main()