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


if __name__ == '__main__':

    #"PLA"
    c, a, b = -2.6294538503860645, -5.541376888553266, 1.3736617277956444
    plot_data("resultstest.txt","PLA", a, b, c)
    
    #"Linear Regression pocket"
    c, a, b =-0.8794538503860644, -1.5901573763582457, 0.02021118320792811
    plot_data("resultstest.txt","Linear Regression pocket", a, b, c)

    #gradient Descent"
    c, a, b =-0.8794538503860644, -1.5901573763582457, 0.02021118320792811
    plot_data("resultstest.txt","gradient Descent", a, b, c)


    j,a,b,c,d,e,f,g,h,i =  -0.9351506061862453, 0.21963155845549406,  1.4184995791657566,  1.8049132223029654, -1.6458597767193168,  -5.309511901378911, -8.177975494056858,  -5.073681265179912,  -2.293770297391901, -1.9781803042303308
    #plot in the form of ax_1+bx_2+cx_1^2+dx_1x_2+ex_2^2+fx_1^3+g(x_1^2)x_2+hx_1(x_2^2)+ix_2^3+j
    plot_data3rd("resultstest.txt","PLA ", a,b,c,d,e,f,g,h,i, j)

    j,a,b,c,d,e,f,g,h,i =-1.1051506061862453, -1.4071977098371877, 0.1279616291868522, 1.5774414792928517, 0.1307587463196966, -0.024866005488650107, 0.7085905743402486, -0.056762984023540845,  0.9727274123379471,  -0.44782318375475766
     #plot in the form of ax_1+bx_2+cx_1^2+dx_1x_2+ex_2^2+fx_1^3+g(x_1^2)x_2+hx_1(x_2^2)+ix_2^3+j
    plot_data3rd("resultstest.txt","Linear Regression Third Order pocket ", a,b,c,d,e,f,g,h,i, j)

    j,a,b,c,d,e,f,g,h,i = 0.2811594305042883, -0.2510146925548312,  -3.245310340419576,  -35.61821111596772,    38.1734488855034,   246.9974509317157,  -2298.798568605831,   4104.034227722335,  -8740.092936569183, -1020.3278315041102
    #plot in the form of ax_1+bx_2+cx_1^2+dx_1x_2+ex_2^2+fx_1^3+g(x_1^2)x_2+hx_1(x_2^2)+ix_2^3+j
    plot_data3rd("resultstest.txt","gradient Descent Third Order", a,b,c,d,e,f,g,h,i, j)

    exit(0)
