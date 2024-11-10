numbers=[1,2,3,4,5,6,7,8,9,10]
dict={number:number*number for number in list(numbers) if (number%2)==1}
print(dict)