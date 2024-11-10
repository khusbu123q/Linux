import functools
def logging(func):
    def wrapper(*arg,**kwargs):
        print(f'calling {func.__name__}with args:{arg},kwargs{kwargs}')
        result=func(*arg,**kwargs)
        print(f'{func.__name__}returned {result}')
        return result
    return wrapper
@logging
def add(a,b):
    return a+b

def main():
    add(2,3)

if __name__ == '__main__':
    main()