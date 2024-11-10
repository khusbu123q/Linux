sentences='hello , how  are you?'
dict_comprehension={sentence:sentence[::-1] for sentence in sentences.split()  }
print(dict_comprehension)