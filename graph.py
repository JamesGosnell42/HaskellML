import numpy as np
import matplotlib.pyplot as plt

def plot_data(filename):
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

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot points
    plt.scatter(x_values[np.array(labels) == 1], y_values[np.array(labels) == 1], color='blue', label='1s', marker='o')
    plt.scatter(x_values[np.array(labels) == 5], y_values[np.array(labels) == 5], color='red', label='5s', marker='x')

    # Plot the line c + ax + by = 0
    c, a, b = -0.9872100065017031,0.04116386224311976,0.0006739910464001116  # Example coefficients
    a1, b1, c1 = 1, 66, -28
    print(a1*a+b1*b+c1*c)
    x_line = np.linspace(np.min(x_values), np.max(x_values), 100)
    y_line = -(a / b) * x_line - (c / b)
    plt.plot(x_line, y_line, color='green', label=f'{a}x + {b}y + {c} = 0')

    # Set the axis limits based on the data range
    plt.xlim(np.min(x_values), np.max(x_values))
    plt.ylim(np.min(y_values), np.max(y_values))

    # Add labels and legend
    plt.title('Scatter Plot of Data Points')
    plt.xlabel('Overlaps')
    plt.ylabel('Symmetry')
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid()
    plt.legend()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_data("results.txt")
    exit(0)
