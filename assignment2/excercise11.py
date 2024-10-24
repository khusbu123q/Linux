def max_min(dict):
    max_value=max(dict.values())
    min_value=min(dict.values())
    return max_value,min_value

def main():
    dict = {"a": 1, "b": 32, "c": 84,"d":-1,"e":54,"f":-6}
    print(max_min(dict))

if __name__ == '__main__':
    main()