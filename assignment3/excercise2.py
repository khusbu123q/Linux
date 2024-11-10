fruits=['mango','kiwi','strawberry','guava','pineapple','mandarin orange']
def count(fruits,vowels):
    vowels = ["a", "e", "i", "o", "u"]
    total=0
    for j in vowels:
        total=fruits.count(j)
    return total
fruits_with_only_two_vowels=[ i for i in fruits if sum(i.count(j) for j in ("a","e","i","o","u"))==2]
print (fruits_with_only_two_vowels)

