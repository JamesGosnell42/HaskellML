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
                    number = int(number_str.strip())
                    
                    # Remove brackets and convert coordinates to a list
                    coords_str = coords_str.strip('[]')
                    x, y = map(float, coords_str.split(','))  # Split by comma and convert to float
                    
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
    misclassified_points = np.sum((labels == 1) & above_line) + np.sum((labels == 5) & ~above_line)
    error_percentage = (misclassified_points / total_points) * 100
    print(f"Error Percentage: {error_percentage:.2f}%")

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot points based on their labels, without changing the original colors or markers
    plt.scatter(x_values[labels == 1], y_values[labels == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[labels == 5], y_values[labels == 5], color='red', label='5s', marker='x')

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

if __name__ == '__main__':
    #Linear Regression no inter
    c, a, b = -0.9872100065017031, 0.04116386224311976, 0.0006739910464001116
    plot_data("results.txt","Linear Regression no inter", a, b, c)
    #"Linear Regression"
    c, a, b = 1.0127899934982967, 1061.0411638622431, 229.7356739910465
    plot_data("results.txt","Linear Regression", a, b, c)
    
    #"Linear Regression pocket"
    c, a, b = -118.9872100065017,  2184.041163862243,  276.0296739910457
    plot_data("results.txt","Linear Regression pocket", a, b, c)


    exit(0)
