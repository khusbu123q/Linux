import time
import functools

def measure_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in: {execution_time:.4f} seconds")
        return result
    return wrapper


@measure_time
def sample_function(delay):
    time.sleep(delay)  # Sleep to simulate a time-consuming task
    return "Function completed"

def main():
    sample_function(5)

if __name__ == '__main__':
    main()
