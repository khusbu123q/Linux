def extract_words(list,k):
    output=[word for word in list if word.startswith(k)]
    return output

def main():
    list=["khusbu","kiwi","bannana","kaushik","kangaroo","apple","dragon fruit","kinow"]
    letter="k"
    print(extract_words(list,letter))

if __name__ == '__main__':
    main()