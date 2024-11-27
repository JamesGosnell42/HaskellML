import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
from scipy.spatial import distance


def safe_log(p):
    result = np.log(1 - np.exp(p))
    if np.isnan(result):
        return 0
    return result

def logistic_error(weights, x, y):
    predictions = np.dot(x, weights)
    log_predictions = np.vectorize(safe_log)(-y * predictions)
    return -np.mean(log_predictions)

def plot_logistic_regression(filename, title, weights):
    x_values = []
    y_values = []
    labels = []

    # Read the file and parse the data
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)
                    number = float(number_str.strip())
                    
                    coords_str = coords_str.strip('[]')
                    one_str, x_str, y_str = map(str.strip, coords_str.split(','))

                    x = float(x_str)
                    y = float(y_str)

                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    # Add intercept term to x_values
    x_values_with_intercept = np.column_stack((np.ones(x_values.shape[0]), x_values, y_values))

    # Calculate logistic error
    error = logistic_error(weights, x_values_with_intercept, labels)
    print(f"Logistic Error: {error:.2f}")

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values[labels == 1], y_values[labels == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[labels == -1], y_values[labels == -1], color='red', label='-1s', marker='x')

    # Plot decision boundary
    x_min, x_max = x_values.min() - 0.5, x_values.max() + 0.5
    y_min, y_max = y_values.min() - 0.5, y_values.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    zz = predict_logistic(weights, np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)

    plt.contour(xx, yy, zz, levels=[0.5], colors='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'{title} (Error: {error:.2f}%)')
    plt.show()

def plot_data(filename, title, a, b, c):
    # Initialize lists to store x and y coordinates
    x_values = []
    y_values = []
    labels = []

    # Read the file and parse the data
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one, x, y= map(str, coords_str.split(','))  # Split by comma and convert to float
                    
                    if(x.find('e') != -1):
                        resx, eval = map(float, x.split('e'))
                        x= resx * 10**eval
                    else: x = float(x)

                    if(y.find('e') != -1):
                        resy, eval = map(float, y.split('e'))
                        y= resy * 10**eval
                    else: y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    # Convert lists to numpy arrays for easy indexing
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    # Determine if each point is above or below the line ax + by + c = 0
    above_line = a * x_values + b * y_values + c > 0

    # Calculate the error percentage
    total_points = len(x_values)
    misclassified_points = np.sum((labels == -1) & above_line) + np.sum((labels == 1) & ~above_line)
    error_percentage = (misclassified_points / total_points) * 100
    print(f"Error Percentage: {error_percentage:.2f}%")

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot points based on their labels, without changing the original colors or markers
    plt.scatter(x_values[labels == 1], y_values[labels == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[labels == -1], y_values[labels == -1], color='red', label='5s', marker='x')

    # Plot the line ax + by + c = 0
    x_line = np.linspace(np.min(x_values), np.max(x_values), 100)
    y_line = -(a / b) * x_line - (c / b)
    plt.plot(x_line, y_line, color='green', label=f'{a}x + {b}y + {c} = 0')

    # Set the axis limits based on the data range
    plt.xlim(np.min(x_values), np.max(x_values))
    plt.ylim(np.min(y_values), np.max(y_values))

    # Add labels and legend
    plt.title(f'{title} (Error: {error_percentage:.2f}%)')
    plt.xlabel('Overlaps')
    plt.ylabel('Symmetry')
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid()
    plt.legend()
    
    # Show the plot
    plt.show()

def plot_data3rd(filename, title, a, b, c, d, e, f, g, h, i, j):
    # Initialize lists to store x and y coordinates
    x_values = []
    y_values = []
    labels = []

    # Read the file and parse the data
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                     # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one, x, y= map(str, coords_str.split(','))  # Split by comma and convert to float
                    
                    if(x.find('e') != -1):
                        resx, eval = map(float, x.split('e'))
                        x= resx * 10**eval
                    else: x = float(x)

                    if(y.find('e') != -1):
                        resy, eval = map(float, y.split('e'))
                        y= resy * 10**eval
                    else: y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    # Convert lists to numpy arrays for easy indexing
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    # Determine if each point satisfies the 3rd order polynomial equation
    above_line = (
        a * x_values + b * y_values +
        c * x_values**2 + d * x_values * y_values + e * y_values**2 +
        f * x_values**3 + g * (x_values**2) * y_values + h * x_values * (y_values**2) + i * y_values**3 +
        j
    ) > 0

    # Calculate the error percentage
    total_points = len(x_values)
    misclassified_points = np.sum((labels == -1) & above_line) + np.sum((labels == 1) & ~above_line)
    error_percentage = (misclassified_points / total_points) * 100
    print(f"Error Percentage: {error_percentage:.2f}%")

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot points based on their labels, without changing the original colors or markers
    plt.scatter(x_values[labels == 1], y_values[labels == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[labels == -1], y_values[labels == -1], color='red', label='5s', marker='x')

    # Generate a finer mesh grid for plotting the decision boundary
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(np.min(x_values), np.max(x_values), 300),  # Increase the density
        np.linspace(np.min(y_values), np.max(y_values), 300)
    )
    z_mesh = (
        a * x_mesh + b * y_mesh +
        c * x_mesh**2 + d * x_mesh * y_mesh + e * y_mesh**2 +
        f * x_mesh**3 + g * (x_mesh**2) * y_mesh + h * x_mesh * (y_mesh**2) + i * y_mesh**3 +
        j
    )

    # Plot the decision boundary
    plt.contour(x_mesh, y_mesh, z_mesh, levels=[0], colors='green')

    # Set the axis limits based on the data range
    plt.xlim(np.min(x_values), np.max(x_values))
    plt.ylim(np.min(y_values), np.max(y_values))

    # Add labels and legend
    plt.title(f'{title} (Error: {error_percentage:.2f}%)')
    plt.xlabel('Overlaps')
    plt.ylabel('Symmetry')
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid()
    plt.legend()
    
    # Show the plot
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_data8th(filename, title, *coefficients):
    # Initialize lists to store x and y coordinates
    x_values = []
    y_values = []
    labels = []

    # Read the file and parse the data
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one, x, y = map(str, coords_str.split(','))  # Split by comma and convert to float
                    
                    if x.find('e') != -1:
                        resx, eval = map(float, x.split('e'))
                        x = resx * 10**eval
                    else:
                        x = float(x)

                    if y.find('e') != -1:
                        resy, eval = map(float, y.split('e'))
                        y = resy * 10**eval
                    else:
                        y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    # Convert lists to numpy arrays for easy indexing
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    # Generate Legendre terms
    terms = generate_legendre_terms(x_values, y_values, max_order=8)

    # Calculate boundary
    decision_values = sum(c * t for c, t in zip(coefficients, terms))
    above_line = decision_values > 0

    # Calculate error percentage
    total_points = len(x_values)
    misclassified_points = np.sum((labels == -1) & above_line) + np.sum((labels == 1) & ~above_line)
    error_percentage = (misclassified_points / total_points) * 100
    print(f"Error Percentage: {error_percentage:.2f}%")

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values[labels == 1], y_values[labels == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[labels == -1], y_values[labels == -1], color='red', label='Others', marker='x')

    # Set plot limits
    x_min, x_max = x_values.min() - 0.5, x_values.max() + 0.5
    y_min, y_max = y_values.min() - 0.5, y_values.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Generate terms for plotting
    terms_plot = generate_legendre_terms(xx.ravel(), yy.ravel(), max_order=8)

    # Compute zz values
    zz = sum(c * t for c, t in zip(coefficients, terms_plot))
    zz = zz.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, zz, levels=[0], colors='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'{title} (Error: {error_percentage:.2f}%)')
    plt.show()


def legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_n_minus_two = np.ones_like(x)
        P_n_minus_one = x
        for k in range(2, n + 1):
            P_n = ((2 * k - 1) * x * P_n_minus_one - (k - 1) * P_n_minus_two) / k
            P_n_minus_two = P_n_minus_one
            P_n_minus_one = P_n
        return P_n_minus_one

def parse_normalized_data(filename):
    x_values = []
    y_values = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one_str, x_str, y_str = map(str.strip, coords_str.split(','))  # Split by comma and strip
                    x,y = x_str, y_str
                    if x_str.find('e') != -1:
                        resx, eval = map(float, x_str.split('e'))
                        x = resx * 10**eval
                    else:
                        x = float(x)

                    if y.find('e') != -1:
                        resy, eval = map(float, y.split('e'))
                        y = resy * 10**eval
                    else:
                        y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    # Convert lists to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    labels = np.array(labels)

    return x_values, y_values, labels

def generate_legendre_terms(x_values, y_values, max_order=8):
    terms = []
    for d in range(max_order + 1):
        if d == 0:
            # Degree 0: [1]
            terms.append(np.ones_like(x_values))
        elif d == 1:
            # Degree 1: [P1(x), P1(y)]
            terms.append(legendre(1, x_values))
            terms.append(legendre(1, y_values))
        else:
            # Degree d >= 2
            terms.append(legendre(d, x_values))  # P_d(x)
            terms.append(legendre(d, y_values))  # P_d(y)
            for i in range(1, d):
                # Cross terms: P_i(x) * P_{d - i}(y)
                term = legendre(i, x_values) * legendre(d - i, y_values)
                terms.append(term)
    return terms
    
def parse_results_full_order(filename):
    transformed_features = []

    with open(filename, 'r') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if line:
                try:
                    # Remove parentheses and split the line into label and features
                    line = line.strip('()')
                    number_str, features_str = line.split(',', 1)
                    number = float(number_str.strip())

                    # Remove brackets and convert features to a list of floats
                    features_str = features_str.strip('[]')
                    feature_strings = features_str.split(',')
                    features = []
                    for f in feature_strings:
                        if f.find('e') != -1:
                            num, eval = map(float, f.split('e'))
                            number = num * 10**eval
                            features.append(number)
                        else:
                            number = float(f)
                            features.append(number)

                    # Verify the number of features
                    if len(features) != 45:
                        print(f"Warning: Line {idx} has {len(features)} features instead of 45. Skipping this line.")
                        continue  # Skip this line

                    transformed_features.append(features)

                except ValueError as e:
                    print(f"Skipping line {idx} due to error: {line} - {e}")
                    continue
    transformed_features = np.array(transformed_features)

    return transformed_features

def compute_legendre_features(x_values, y_values, max_order=8):
    terms = []
    
    for d in range(max_order + 1):  # Degrees from 0 to max_order inclusive
        if d == 0:
            # Degree 0: [1]
            term = np.ones_like(x_values)  # legendre(0, x_values) == 1
            terms.append(term)
        elif d == 1:
            # Degree 1: [P1(x), P1(y)]
            terms.append(legendre(1, x_values))
            terms.append(legendre(1, y_values))
        else:
            # Degree d >= 2
            # Add P_d(x) and P_d(y)
            terms.append(legendre(d, x_values))
            terms.append(legendre(d, y_values))
            # Cross terms: P_i(x) * P_{d - i}(y) for i from 1 to d - 1
            for i in range(1, d):
                P_i_x = legendre(i, x_values)
                P_d_minus_i_y = legendre(d - i, y_values)
                term = P_i_x * P_d_minus_i_y
                terms.append(term)
    
    # Stack terms column-wise
    features = np.column_stack(terms)  # Shape: (num_points, num_features)
    return features

def compare_features(computed_features, haskell_features, tolerance=1e-6):
    discrepancies = []
    num_points, num_features = computed_features.shape

    for idx in range(num_points):
        python_features = computed_features[idx]
        haskell_features_point = haskell_features[idx]

        # Compare features element-wise
        differences = np.abs(python_features - haskell_features_point)
        if not np.allclose(python_features, haskell_features_point, atol=tolerance):
            discrepancies.append((idx, differences))

    return discrepancies

def parse_input(input_string):
    # Split the input string by whitespace to get individual number strings
    number_strings = input_string.split()
    
    # Convert each number string to a float, handling scientific notation
    numbers = []
    for num_str in number_strings:
        try:
            # Convert the string to a float
            if num_str.find('e') != -1:
                num, eval = map(float, num_str.split('e'))
                number = num * 10**eval
            else:
                number = float(num_str)

            numbers.append(number)
        except ValueError as e:
            print(f"Skipping invalid number: {num_str} - {e}")
    
    return np.array(numbers)

def parse_linput(input_string):
    # Split the input string by whitespace to get individual number strings
    number_strings = input_string.split(',')
    
    # Convert each number string to a float, handling scientific notation
    numbers = []
    for num_str in number_strings:
        try:
            # Convert the string to a float
            if num_str.find('e') != -1:
                num, eval = map(float, num_str.split('e'))
                number = num * 10**eval
            else:
                number = float(num_str)

            numbers.append(number)
        except ValueError as e:
            print(f"Skipping invalid number: {num_str} - {e}")
    
    return np.array(numbers)

def plot_errors(glambda, cverr, testerr):
    # Convert lists to numpy arrays for easy indexing
    x_values = np.array(glambda)
    y1_values = np.array(cverr)
    y2_values = np.array(testerr)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y1_values, color='blue', label='CV errors', marker='o')
    plt.plot(x_values, y2_values, color='red', label='Test Errors', marker='o')

    # Set plot limits
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title('CV Errors and Test Errors')
    plt.legend()
    plt.grid(True)
    plt.show()


def parse_and_plot_points(points_str, title):
    # Remove parentheses and split the string into individual points
    points_str = points_str.strip('()')
    points_list = points_str.split('),(')
    
    # Initialize lists to store x and y coordinates
    x_values = []
    y_values = []

    # Parse each point and extract x and y coordinates
    for point in points_list:
        x_str, y_str = point.split(',')
        x_values.append(float(x_str))
        y_values.append(float(y_str))

    # Plot the points
    plt.scatter(x_values, y_values, color='blue', label='Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def parse_file(filename):
    x_values = []
    y_values = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one_str, x_str, y_str = map(str.strip, coords_str.split(','))  # Split by comma and strip
                    x, y = x_str, y_str
                    if x_str.find('e') != -1:
                        resx, eval = map(float, x_str.split('e'))
                        x = resx * 10**eval
                    else:
                        x = float(x)

                    if y.find('e') != -1:
                        resy, eval = map(float, y.split('e'))
                        y = resy * 10**eval
                    else:
                        y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    return np.array(x_values), np.array(y_values), np.array(labels)

import matplotlib.pyplot as plt
import numpy as np

def parse_file(filename):
    x_values = []
    y_values = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                try:
                    # Remove parentheses and split the line into number and coordinates
                    line = line.strip('()')
                    number_str, coords_str = line.split(',', 1)  # Split at the first comma
                    number = float(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    one_str, x_str, y_str = map(str.strip, coords_str.split(','))  # Split by comma and strip
                    x, y = x_str, y_str
                    if x_str.find('e') != -1:
                        resx, eval = map(float, x_str.split('e'))
                        x = resx * 10**eval
                    else:
                        x = float(x)

                    if y.find('e') != -1:
                        resy, eval = map(float, y.split('e'))
                        y = resy * 10**eval
                    else:
                        y = float(y)

                    # Append the values to the respective lists
                    x_values.append(x)
                    y_values.append(y)
                    labels.append(number)
                except ValueError as e:
                    print(f"Skipping line due to error: {line} - {e}")
                except SyntaxError as e:
                    print(f"Skipping line due to error in format: {line} - {e}")

    return np.array(x_values), np.array(y_values), np.array(labels)

def parse_input(input_str):
    lines = input_str.strip().split('\n')
    centers = []
    weights = []
    sigma = None

    parsing_centers = False
    parsing_weights = False
    for line in lines:
        line = line.strip()
        if line.startswith('centers = ['):
            parsing_centers = True
            parsing_weights = False
            continue
        elif line.startswith('weights = ['):
            parsing_centers = False
            parsing_weights = True
            continue
        elif line.startswith('sigma = '):
            sigma = float(line.split('=')[1].strip())
            continue
        elif parsing_centers and line.startswith('['):
            values = line.strip('[],').split(',')
            parsed_values = []
            for value in values:
                value = value.strip()  # Remove leading and trailing spaces
                if 'e' in value:
                    num, exp = map(float, value.split('e'))
                    parsed_values.append(num * 10**exp)
                else:
                    parsed_values.append(float(value))
            centers.append(parsed_values[1:])  # Remove the bias variable
        elif parsing_weights and line and line != ']':
            value = line.strip().replace(',', '')  # Remove commas
            if 'e' in value:
                num, exp = map(float, value.split('e'))
                weights.append(num * 10**exp)
            else:
                weights.append(float(value))

    return np.array(centers), np.array(weights).flatten(), sigma

def rbf_kernel(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * sigma**2))

def compute_rbf_kernel(centers, features, sigma):
    kernel_matrix = np.zeros((features.shape[0], centers.shape[0]))
    for i, feature in enumerate(features):
        for j, center in enumerate(centers):
            kernel_matrix[i, j] = rbf_kernel(feature, center, sigma)
    return kernel_matrix

def rbf_classify(test_points, centers, weights, sigma):
    kernel_matrix = compute_rbf_kernel(centers, test_points, sigma)
    predictions = np.dot(kernel_matrix, weights)
    return np.where(predictions > 0, 1, -1)  # Classify as 1 if prediction >= 0, else -1

def classify_and_plot_rbf(data_file, centers, weights, sigma):
    x_values, y_values, labels = parse_file(data_file)
    test_points = np.column_stack((x_values, y_values))

    predictions = rbf_classify(test_points, centers, weights, sigma)

    # Calculate the error
    error = np.mean(predictions != labels)
    print(f"Classification error: {error}")

    # Plot the data with predictions
    colors = np.where(predictions == labels, predictions, 0)  # Use 0 for misclassified points
    plt.scatter(x_values, y_values, c=colors, cmap='coolwarm', marker='x', label='RBF Predictions')
    plt.colorbar(label='Prediction Value')

    # Plot the centers with their weight radius
    centers_x = centers[:, 0]
    centers_y = centers[:, 1]
    for i, (cx, cy) in enumerate(zip(centers_x, centers_y)):
        plt.scatter(cx, cy, c='black', marker='o')
        circle = plt.Circle((cx, cy), sigma, color='black', fill=False)
        plt.gca().add_artist(circle)
        plt.scatter(cx, cy, c='blue' if weights[i] > 0 else 'red', marker='o', edgecolor='black')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RBF Classification')
    plt.legend()
    plt.grid(True)
    plt.show()

def knn_classify(test_points, train_points, train_labels, k):
    predictions = []
    for test_point in test_points:
        distances = distance.cdist([test_point], train_points, 'euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        prediction = np.sign(np.sum(nearest_labels))
        predictions.append(prediction)
    return np.array(predictions)

def classify_and_plot_knn(training_file, testing_file, k):
    # Parse the training data
    train_x, train_y, train_labels = parse_file(training_file)
    train_points = np.column_stack((train_x, train_y))

    # Parse the testing data
    test_x, test_y, test_labels = parse_file(testing_file)
    test_points = np.column_stack((test_x, test_y))

    # Classify the test points
    predictions = knn_classify(test_points, train_points, train_labels, k)

    # Calculate the error
    error = np.mean(predictions != test_labels)
    print(f"Classification error: {error}")

    # Plot the data with predictions
    colors = np.where(predictions == test_labels, predictions, 0)  # Use 0 for misclassified points
    plt.scatter(test_x, test_y, c=colors, cmap='coolwarm', marker='x', label='kNN Predictions')
    plt.colorbar(label='Prediction Value')

    # Plot the training points
    plt.scatter(train_x, train_y, c=train_labels, cmap='coolwarm', marker='o', edgecolor='black', label='Training Points')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('kNN Classification')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # File names
    normalized_data_file = 'normalizedData.txt'
    normalized_test_data_file = 'normalizedDataTest.txt'

    points_str = "(1,2.4545454545454546),(2,2.757575757575758),(3,1.8484848484848486),(4,2.757575757575758),(5,3.060606060606061),(6,2.757575757575758),(7,3.0303030303030307),(8,3.333333333333334),(9,3.333333333333334),(10,3.93939393939394),(11,4.242424242424243),(12,4.242424242424243),(13,4.545454545454546),(14,4.848484848484849),(15,4.848484848484849),(16,5.151515151515152),(17,5.151515151515152),(18,5.151515151515152),(19,5.454545454545455),(20,5.757575757575759),(21,6.666666666666668),(22,7.575757575757577),(23,7.87878787878788),(24,7.87878787878788),(25,7.87878787878788),(26,7.87878787878788),(27,7.87878787878788),(28,8.484848484848486),(29,8.484848484848486),(30,8.787878787878789),(31,8.787878787878789),(32,9.090909090909092),(33,9.090909090909092),(34,9.090909090909092),(35,9.090909090909092),(36,9.696969696969695),(37,10.0),(38,10.0),(39,10.0),(40,10.0),(41,10.909090909090908),(42,11.212121212121211),(43,11.212121212121211),(44,11.212121212121211),(45,11.515151515151514),(46,11.515151515151514),(47,11.818181818181817),(48,12.12121212121212),(49,12.424242424242424),(50,13.03030303030303)"
    parse_and_plot_points(points_str, "kNN k vs Error")

    classify_and_plot_knn(normalized_data_file, normalized_test_data_file, 3)

    points_str = "(1,17.909090909090903),(3,10.93939393939394),(5,10.93939393939394),(7,7.272727272727274),(9,7.303030303030304),(11,6.666666666666668),(13,6.696969696969698),(15,6.3939393939393945),(17,7.272727272727274),(19,6.060606060606061),(21,7.272727272727274),(23,7.575757575757577),(25,7.272727272727274),(27,6.969696969696971),(29,6.666666666666668),(31,7.303030303030304),(33,4.545454545454546),(35,5.151515151515152),(37,5.454545454545455),(39,4.848484848484849),(41,5.151515151515152),(43,4.545454545454546),(45,5.757575757575759),(47,3.636363636363637),(49,3.93939393939394),(51,4.242424242424243),(53,3.333333333333334),(55,3.636363636363637),(57,4.242424242424243),(59,3.93939393939394),(61,5.151515151515152),(63,3.93939393939394),(65,4.545454545454546),(67,6.060606060606061),(69,5.151515151515152),(71,5.757575757575759),(73,5.757575757575759),(75,5.454545454545455),(77,5.757575757575759),(79,7.272727272727274),(81,6.060606060606061),(83,8.484848484848486),(85,7.87878787878788),(87,5.454545454545455),(89,6.666666666666667),(91,7.606060606060607),(93,8.212121212121213),(95,8.212121212121213),(97,7.90909090909091),(99,7.606060606060607)"
    parse_and_plot_points(points_str, "RBF k vs Error")

  
    filename = "normalizedData.txt"  # Replace with your actual filename
    title = "8th Order Legendre Polynomial Decision Boundary lamba = 0 "
    coefficients = parse_input("11.885986407358683   -38.98645952415889   35.364207491196986   48.525624273337236    24.65655286608682   -77.26115743456144  -38.631116844864835   24.519825630906897   -93.75271149681267    107.5597564502562    41.24842958193097   0.8668471797681239   -30.89578764081576    93.78745473112731  -111.14221642416803   -9.880900548070969   2.6518409292660086  -27.086618532679065     70.3557883511933   -86.53224891136614    51.53562597089382   12.534306185949312 -0.48192857036008163    4.703775482249351    4.408145145685502   -46.70363078421796    80.53577475408554  -58.304210647118154 -0.24687191226384073  -0.2661635178705546 9.734415714032885e-3    7.360715689958498  -19.768273778208453   24.355416288689742   -17.16458455535486    6.148666430470627   0.8646608717567256  0.20137576563892512 -0.21766378133032818  -2.3314872129213917    3.552110068267951    7.309827081770621  -22.272581048044962   20.794667538287516    -8.04337400117317")
    #plot_data8th(filename, title, *coefficients)
    #plot_data8th(normalized_test_data_file, title, *coefficients)

    input_str = """
    centers = [
        [1.0, 43.333333333333336, -97.86844444444445],
        [1.0, 45.0, 37.805],
        [1.0, 3.25, -208.45725000000002],
        [1.0, 26.0, -34.3825],
        [1.0, 58.0, -163.0765],
        [1.0, 4.5, -152.29825],
        [1.0, 65.0, -66.52250000000001],
        [1.0, 64.66666666666667, -123.39033333333333],
        [1.0, 64.0, -22.933],
        [1.0, 40.0, -206.58100000000002],
        [1.0, 34.0, -59.757857142857155],
        [1.0, 32.8421052631579, -132.26268421052632],
        [1.0, 23.583333333333332, -174.91141666666667],
        [1.0, 33.5, -1.8619999999999992],
        [1.0, 17.25, -117.8645],
        [1.0, 60.333333333333336, -101.29050000000001],
        [1.0, 3.8333333333333335, -179.8088333333333],
        [1.0, 55.4, -47.394999999999996],
        [1.0, 54.8, -83.9876],
        [1.0, 25.25, -160.27325],
        [1.0, 27.916666666666668, -98.82616666666668],
        [1.0, 62.75, -142.801],
        [1.0, 41.0, -163.32779999999997],
        [1.0, 17.666666666666668, -132.42583333333334],
        [1.0, 18.0, -201.96200000000002],
        [1.0, 52.0, -113.83749999999999],
        [1.0, 37.666666666666664, -83.66011111111112],
        [1.0, 51.5, -8.711500000000001],
        [1.0, 71.0, -44.141999999999996],
        [1.0, 84.0, -135.925],
        [1.0, 26.444444444444443, -122.22677777777777],
        [1.0, 26.666666666666668, -51.27366666666668],
        [1.0, 49.0, -60.31733333333333],
        [1.0, 10.875, -166.58049999999997],
        [1.0, 43.111111111111114, -46.059111111111115],
        [1.0, 80.0, -75.90899999999999],
        [1.0, 51.666666666666664, -25.74366666666667],
        [1.0, 33.9, -147.60375000000002],
        [1.0, 49.5, -129.51725],
        [1.0, 76.0, -106.97000000000001],
        [1.0, 25.0, -75.64700000000002],
        [1.0, 17.25, -188.345125],
        [1.0, 43.0, -12.310999999999998],
        [1.0, 3.6, -137.7858],
        [1.0, 24.0, -213.7955],
        [1.0, 47.2, -143.2878],
        [1.0, 48.333333333333336, -70.892],
        [1.0, 2.2, -193.98829999999998],
        [1.0, 22.875, -145.12775],
        [1.0, 49.666666666666664, -104.40883333333333],
        [1.0, 38.95, -116.32369999999999],
        [1.0, 16.333333333333332, -104.49166666666666],
        [1.0, 68.0, -88.537]
    ]
    weights = [
        -9.372564251714353e-23,
        -0.9900990099009901,
        2.9383352329661204e-7,
        -1.3233090383683663e-59,
        -1.5548822231700656e-44,
        8.488517062026785e-24,
        -1.9728780639153856e-91,
        -9.92745512961038e-38,
        -0.9900990099009901,
        -0.9900990099009901,
        -0.9900991411156261,
        -1.4673558209114602e-3,
        -8.322383613795612e-7,
        -6.535886271140841e-87,
        9.427428387553828e-25,
        -4.006853216013094,
        7.767107568514572e-4,
        -5.64338486686404e-17,
        -7.3454431822012065e-40,
        -5.812365919437675e-4,
        -2.0798461361787455e-2,
        -1.038142709072964,
        -2.8974935519327173e-25,
        -1.9802017196632243e-8,
        -0.9900990099009901,
        -3.562499057344893e-2,
        -3.331272207968721e-19,
        -5.212076350214798e-36,
        -0.9900990099009901,
        -0.9900990099009901,
        -2.6713466428709948e-36,
        -1.4596607766825756e-19,
        -2.4474972228077383e-27,
        1.5538952456371482,
        -0.3863970636206811,
        -0.9900990099009901,
        -4.86472770764573e-13,
        -0.35910066983387423,
        -5.3416569274259295e-8,
        -0.9900990099009901,
        -3.3421451270938507e-16,
        -2.520350909637586e-3,
        -1.6736533260769867e-12,
        0.15700155196260943,
        -6.922268701072305e-71,
        -7.374808530228488e-77,
        -1.8126544812021938e-40,
        2.8617444479971214e-2,
        -5.905705146712817e-4,
        -3.4303961962213325,
        -7.744474208422625e-8,
        -7.392683052283323e-28,
        -0.9900990099009901
    ]
    sigma = 0.27472112789737807
    """
    centers, weights, sigma = parse_input(input_str)
    print(centers)
    print(weights)
    print(sigma)
    classify_and_plot_rbf(normalized_test_data_file, centers, weights, sigma)

    exit(0)
