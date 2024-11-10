class FormulaError(Exception):
    """Custom exception to handle formula errors in the calculator."""
    pass


def calculate(user_input):
    part = user_input.split()

    # Check if the input has exactly 3 parts
    if len(part) != 3:
        raise FormulaError("Input does not consist of three elements")

    num1, operator, num2 = part


    try:
        num1 = float(num1)
        num2 = float(num2)
    except ValueError:
        raise FormulaError("The first and third elements must be numbers")


    if operator not in ('+', '-'):
        raise FormulaError("The operator must be '+' or '-'")


    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2


def main():
    while True:
        user_input = input("Enter a formula (or type 'quit' to exit): ")

        if user_input.lower() == "quit":
            print("Calculator exiting...")
            break

        try:
            result = calculate(user_input)
            print("Result:", result)
        except FormulaError as e:
            print("Formula error:", e)


if __name__ == '__main__':
    main()
