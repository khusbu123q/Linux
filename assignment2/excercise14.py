def remove(sentence):
    dict={}
    word_list=sentence.split(" ")
    unique_words=[]
    for word in word_list:
        if word not in dict:
            dict[word]=True
            unique_words.append(word)
    result_sentence="".join(unique_words)
    return result_sentence

def main():
    sen=input("enter the string")
    remove(sen)
    print(remove(sen))

if __name__ == '__main__':
    main()
