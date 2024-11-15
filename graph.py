import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
  # File names
    normalized_data_file = 'normalizedData.txt'
    normalized_test_data_file = 'normalizedTestData.txt'
    c, a, b = -3.854440869674106, -6.433939164555101, 1.6543833824952823
    plot_data(normalized_data_file, "Linear Regression 2nd order", a, b, c)
    plot_data(normalized_test_data_file, "Linear Regression 2nd order", a, b, c)
    c, a, b =-0.9244408696741061,   -1.2191504321607296, 0.024335637065230866
    plot_data(normalized_data_file, "Linear Regression 2nd order", a, b, c)
    plot_data(normalized_test_data_file, "Linear Regression 2nd order", a, b, c)
    filename = "normalizedData.txt"  # Replace with your actual filename
    title = "8th Order Legendre Polynomial Decision Boundary lamba = 0 "
    coefficients = parse_input("11.885986407358683   -38.98645952415889   35.364207491196986   48.525624273337236    24.65655286608682   -77.26115743456144  -38.631116844864835   24.519825630906897   -93.75271149681267    107.5597564502562    41.24842958193097   0.8668471797681239   -30.89578764081576    93.78745473112731  -111.14221642416803   -9.880900548070969   2.6518409292660086  -27.086618532679065     70.3557883511933   -86.53224891136614    51.53562597089382   12.534306185949312 -0.48192857036008163    4.703775482249351    4.408145145685502   -46.70363078421796    80.53577475408554  -58.304210647118154 -0.24687191226384073  -0.2661635178705546 9.734415714032885e-3    7.360715689958498  -19.768273778208453   24.355416288689742   -17.16458455535486    6.148666430470627   0.8646608717567256  0.20137576563892512 -0.21766378133032818  -2.3314872129213917    3.552110068267951    7.309827081770621  -22.272581048044962   20.794667538287516    -8.04337400117317")
    plot_data8th(filename, title, *coefficients)
    plot_data8th(normalized_test_data_file, title, *coefficients)

    exit(0)
