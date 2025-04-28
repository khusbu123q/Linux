import math

# Solve quadratic equation ax^2 + bx + c = 0
def solve_quadratic_equation(coeff_a, coeff_b, coeff_c):
    discriminant = coeff_b ** 2 - 4 * coeff_a * coeff_c
    if discriminant < 0:
        return None
    sqrt_discriminant = math.sqrt(discriminant)
    root1 = (-coeff_b + sqrt_discriminant) / (2 * coeff_a)
    root2 = (-coeff_b - sqrt_discriminant) / (2 * coeff_a)
    return root1, root2

# Manually multiply a matrix with a vector
def multiply_matrix_vector(matrix, vector):
    product = []
    for row in matrix:
        row_sum = sum(row[i] * vector[i] for i in range(len(vector)))
        product.append(row_sum)
    return product

# Compute quadratic form: xᵀ A x
def compute_quadratic_form(matrix, vector):
    intermediate_product = multiply_matrix_vector(matrix, vector)
    quadratic_sum = sum(vector[i] * intermediate_product[i] for i in range(len(vector)))
    return quadratic_sum

# Check positive definiteness using quadratic form
def check_definiteness_by_quadratic_form(matrix):
    test_vector = [1, 1]  # Example non-zero vector
    quad_form_result = compute_quadratic_form(matrix, test_vector)
    if quad_form_result > 0:
        return "The matrix is Positive Definite"
    elif quad_form_result == 0:
        return "The matrix is Positive Semi-Definite"
    else:
        return "The matrix is Negative Definite"

# Check positive definiteness using leading principal minors
def is_matrix_positive_definite(matrix):
    leading_minor_1 = matrix[0][0]
    determinant_full = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    print(f"Leading Minor: {leading_minor_1}, Determinant: {determinant_full}")
    return leading_minor_1 > 0 and determinant_full > 0

# Find eigenvalues of a 2x2 matrix
def compute_eigenvalues(matrix):
    element_00, element_01 = matrix[0]
    element_10, element_11 = matrix[1]
    coeff_a = 1
    coeff_b = -(element_00 + element_11)
    coeff_c = element_00 * element_11 - element_01 * element_10
    eigenvalues = solve_quadratic_equation(coeff_a, coeff_b, coeff_c)
    return eigenvalues

# Hessian of f(x,y) = x^3 + 2y^3 - xy
def hessian_first_function(x_val, y_val):
    return [
        [6 * x_val, -1],
        [-1, 12 * y_val]
    ]

# Hessian of f(x,y) = 4x + 2y - x^2 - 3y^2
def hessian_second_function():
    return [
        [-2, 0],
        [0, -6]
    ]

if __name__ == "__main__":

    # 1. Positive definiteness of matrix A
    matrix_A = [[9, -15], [-15, 21]]
    if is_matrix_positive_definite(matrix_A):
        print("Matrix A is Positive Definite.")
    else:
        print("Matrix A is NOT Positive Definite (possibly Indefinite).")

    # 2. Positive definiteness using quadratic form
    definiteness_result = check_definiteness_by_quadratic_form(matrix_A)
    print(definiteness_result)

    # 3. Eigenvalues of Hessian matrix at point (3,1)
    hessian_at_3_1 = [[108, -1], [-1, 2]]
    eigenvalues_at_3_1 = compute_eigenvalues(hessian_at_3_1)
    print(f"Eigenvalues at (3,1): {eigenvalues_at_3_1}")

    # 4. Concavity analysis for f(x,y) = x^3 + 2y^3 - xy
    print("\nConcavity analysis for f(x,y) = x^3 + 2y^3 - xy")

    # (i) At (0,0)
    print("\nAt (0,0):")
    hessian_0_0 = hessian_first_function(0, 0)
    eigen_0_0 = compute_eigenvalues(hessian_0_0)
    print(f"Eigenvalues: {eigen_0_0}")
    if eigen_0_0[0] * eigen_0_0[1] < 0:
        print("Saddle point.")
    elif eigen_0_0[0] > 0:
        print("Local Minimum.")
    else:
        print("Local Maximum.")

    # (ii) At (3,3)
    print("\nAt (3,3):")
    hessian_3_3 = hessian_first_function(3, 3)
    eigen_3_3 = compute_eigenvalues(hessian_3_3)
    print(f"Eigenvalues: {eigen_3_3}")
    if eigen_3_3[0] > 0 and eigen_3_3[1] > 0:
        print("Local Minimum.")
    elif eigen_3_3[0] < 0 and eigen_3_3[1] < 0:
        print("Local Maximum.")
    else:
        print("Saddle point.")

    # (iii) At (3,-3)
    print("\nAt (3,-3):")
    hessian_3_minus3 = hessian_first_function(3, -3)
    eigen_3_minus3 = compute_eigenvalues(hessian_3_minus3)
    print(f"Eigenvalues: {eigen_3_minus3}")
    if eigen_3_minus3[0] > 0 and eigen_3_minus3[1] > 0:
        print("Local Minimum.")
    elif eigen_3_minus3[0] < 0 and eigen_3_minus3[1] < 0:
        print("Local Maximum.")
    else:
        print("Saddle point.")

    # 5. Find critical point and classify for f(x,y) = 4x + 2y - x^2 - 3y^2
    print("\nCritical point and classification for f(x,y) = 4x + 2y - x^2 - 3y^2")
    critical_x = 2
    critical_y = 1 / 3
    print(f"Critical point: ({critical_x}, {critical_y})")

    hessian_second = hessian_second_function()
    eigen_second = compute_eigenvalues(hessian_second)
    print(f"Eigenvalues: {eigen_second}")

    if eigen_second[0] > 0 and eigen_second[1] > 0:
        print("Local Minimum.")
    elif eigen_second[0] < 0 and eigen_second[1] < 0:
        print("Local Maximum.")
    else:
        print("Saddle point.")

