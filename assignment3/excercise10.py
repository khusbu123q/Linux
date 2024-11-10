def division(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
        result = None
    except ValueError:
        print("Error: Invalid input. Please provide numeric values.")
        result = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result = None
    else:
        print(f"The result is: {result}")
    finally:
        print("Execution completed.")
    return result
def main():
    division(2,3)
    division(5,8)
    division(0,2)
    division(4,0)
    division("12",5)
if __name__ == '__main__':
    main()