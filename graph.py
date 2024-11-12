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
    results_full_order_file = 'resultsFullOrder.txt'

    # Parse normalized data
    x_values, y_values, labels = parse_normalized_data(normalized_data_file)

    # Compute Legendre features
    computed_features = compute_legendre_features(x_values, y_values, max_order=8)

    # Parse Haskell-generated features
    haskell_features = parse_results_full_order(results_full_order_file)

    # Compare features
    discrepancies = compare_features(computed_features, haskell_features)

    if discrepancies:
        print(f"Found discrepancies in {len(discrepancies)} out of {len(x_values)} data points.")
        for idx, diffs in discrepancies:
            print(f"\nData point index: {idx}")
            print(f"Label: {labels[idx]}")
            print(f"Python features: {computed_features[idx]}")
            print(f"Haskell features: {haskell_features[idx]}")
            print(f"Differences: {diffs}")
    else:
        print("All features match between Python and Haskell implementations.")

    filename = "normalizedData.txt"  # Replace with your actual filename
    title = "8th Order Legendre Polynomial Decision Boundary lamba = 0 "
    coefficients = parse_input(" 107.64357156982152  -265.07688102778985    242.9262784719119    313.0169781690574    243.9315474747367   -617.2864309642348  -237.68491017349874   147.06102154361264   -573.7771476537756     683.018792606852    144.3237954786628  72.92413334762125  -370.34674177113044    657.1226556968799   -532.0735537985471   -65.69873071188931    23.42292147530946  -152.88249927098502    362.0020817680982   -440.3397916625042   290.94656388486914   22.775396291937774     8.51313566677488   -66.35714441329077    164.4985365177007  -233.19097852228919   240.94028095277395  -140.82979346467167  -3.6192648425564755  0.24513955276347194  -16.449010576776754    46.09266032149041   -60.36689929785449    88.39219495772525   -91.00144975866226   37.216806349709515   0.8149452366135761 -0.21652687985236868   -5.488574003648521    17.80963814111545    -17.0250299618423   21.217906476417294   -33.85109489330557   25.408079490991508   -8.242242038279405")
    plot_data8th(filename, title, *coefficients)

    filename = "normalizedData.txt"  # Replace with your actual filename
    title = "8th Order Legendre Polynomial Decision Boundary lamba = 2 "
    coefficients = parse_input("-0.6866395747295059    -0.6069346559252042  1.2067168979996833e-2     0.6256546042730369   8.063668749296976e-2 0.2172718365691859    -0.3258543797004172   8.291594332105494e-2   8.316325746257404e-2   -0.31105029009329943  -6.394216107226884e-2  -8.859915652584688e-2    -0.1682606665633455   4.051799417434346e-2    0.24360012544327603    0.28448745761274147  -8.354455264651317e-2    0.12752470625829282    0.11050100089066395   -0.14882331733249066   -0.11518604322128931    -0.2737561251837606  1.4797286229399301e-2   1.008909257652051e-2  -9.731846796894424e-2   6.504873962809643e-2    0.21992205675705453 -5.7004779545428055e-2    0.12427531070802379  -5.052794642226984e-2  -8.451228297831379e-2   7.292786329858847e-2  1.7166410192412753e-3   -0.22199489523052854   -0.14567433315726686     0.2028941331343631    0.12950387242688108  2.9455201543560225e-2   7.523143777455221e-2   6.802166547051934e-2  -7.365805260129563e-2  5.1162103853479746e-2     0.1855851799382892    7.12101137294514e-3    -0.1099611199807278")
    plot_data8th(filename, title, *coefficients)

    lambdas = parse_linput("0.0,0.1,0.2,0.30000000000000004,0.4,0.5,0.6000000000000001,0.7000000000000001,0.8,0.9,1.0,1.1,1.2000000000000002,1.3,1.4000000000000001,1.5,1.6,1.7000000000000002,1.8,1.9000000000000001,2.0,2.1,2.2,2.3000000000000003,2.4000000000000004,2.5,2.6,2.7,2.8000000000000003,2.9000000000000004,3.0,3.1,3.2,3.3000000000000003,3.4000000000000004,3.5,3.6,3.7,3.8000000000000003,3.9000000000000004,4.0")
    cverrors = parse_linput("2.0,0.6666666666666666,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.3333333333333333,0.6666666666666666,0.6666666666666666,0.6666666666666666")
    testerrors = parse_linput("2.5116692598355193,1.2002667259390976,1.1669259835519004,1.1669259835519004,1.1669259835519004,1.1446988219604357,1.1446988219604357,1.1446988219604357,1.1446988219604357,1.1446988219604357,1.1446988219604357,1.1224716603689708,1.1224716603689708,1.1224716603689708,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1113580795732385,1.1002444987775062,1.1002444987775062,1.1113580795732385,1.1002444987775062,1.1002444987775062,1.1002444987775062,1.0891309179817736,1.1002444987775062,1.0891309179817736,1.0780173371860413,1.0780173371860413,1.0780173371860413,1.066903756390309,1.0557901755945767,1.0446765947988441,1.0446765947988441,1.0335630140031118,1.0224494332073795,1.011335852411647")
    plot_errors(lambdas, cverrors, testerrors)

    filename = "normalizedData.txt"  # Replace with your actual filename
    title = "8th Order Legendre Polynomial Decision Boundary best lamba (3.7)"
    coefficients = parse_input("-0.6915843761921722    -0.6003522424263605  -7.045658762525775e-2     0.6658896668549634   5.901761132218967e-2 0.20648017849650727   -0.35245857070422704  6.0338886837981595e-2    0.11642827431496983    -0.2912875079287152 -1.6255629581621942e-2   2.201594578244405e-3   -0.13896924121347884   3.420300086709892e-2    0.25834624479449053    0.24163397249826662  -6.694110861665115e-2   6.692926932694124e-2     0.1332959289274971   -0.12601787060110353   -0.10321892198788585    -0.2880208041672969   3.246227817440361e-2    7.43969275788275e-2   -0.11832543661541056  -3.482779855953439e-2    0.17100818018247207  -4.737488127553462e-2    0.16207595044582254 -2.9313859662595268e-2  -8.768831070533858e-2 -1.5922406151217386e-2   6.404754751411469e-2   -0.11810078188831453  -7.507428434821635e-2    0.19306399945678063   1.091591482190948e-2   8.773040035788063e-3   4.403947059016645e-2   9.915773856214213e-2  -5.782952629285802e-2 -3.3367295365707546e-2    0.13582122484281012 -1.5163024475633897e-2   -0.13640094537790212")
    plot_data8th(filename, title, *coefficients)

    exit(0)
