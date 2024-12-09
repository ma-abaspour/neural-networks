
import random

def generate_synthetic_data(num_samples=100, slope=3, intercept=4, noise_level=1.0):
    X = [random.uniform(0, 10) for _ in range(num_samples)]
    y = [intercept + slope * x + random.gauss(0, noise_level) for x in X]
    return X, y

def compute_cost(X, y, theta0, theta1):
    m = len(X)
    total_error = 0.0
    for i in range(m):
        prediction = theta0 + theta1 * X[i]
        total_error += (prediction - y[i]) ** 2
    return total_error / (2 * m)

def gradient_descent(X, y, theta0_init, theta1_init, learning_rate, num_iterations):
    theta0 = theta0_init
    theta1 = theta1_init
    m = len(X)
    cost_history = []

    for iteration in range(num_iterations):
        sum_errors_theta0 = 0.0
        sum_errors_theta1 = 0.0
        for i in range(m):
            prediction = theta0 + theta1 * X[i]
            error = prediction - y[i]
            sum_errors_theta0 += error
            sum_errors_theta1 += error * X[i]
        
        theta0 -= (learning_rate / m) * sum_errors_theta0
        theta1 -= (learning_rate / m) * sum_errors_theta1

        cost = compute_cost(X, y, theta0, theta1)
        cost_history.append(cost)

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Cost {cost:.4f}, theta0 {theta0:.4f}, theta1 {theta1:.4f}")

    return theta0, theta1, cost_history

def plot_linear_regression(X, y, theta0, theta1, width=500, height=500, margin=50):
    rgb_matrix = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    x_min = min(X)
    x_max = max(X)
    y_min = min(y)
    y_max = max(y)
    
    def map_x(x):
        return margin + int((x - x_min) / (x_max - x_min) * (width - 2 * margin))
    
    def map_y(y_val):
        return height - margin - int((y_val - y_min) / (y_max - y_min) * (height - 2 * margin))
    
    for xi, yi in zip(X, y):
        px = map_x(xi)
        py = map_y(yi)
        if 0 <= px < width and 0 <= py < height:
            rgb_matrix[py][px] = (0, 0, 255)
    
    x1, y1 = x_min, theta0 + theta1 * x_min
    x2, y2 = x_max, theta0 + theta1 * x_max
    px1, py1 = map_x(x1), map_y(y1)
    px2, py2 = map_x(x2), map_y(y2)
    
    def draw_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < width and 0 <= y < height:
                    rgb_matrix[y][x] = (255, 0, 0)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
            if 0 <= x < width and 0 <= y < height:
                rgb_matrix[y][x] = (255, 0, 0)
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < width and 0 <= y < height:
                    rgb_matrix[y][x] = (255, 0, 0)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
            if 0 <= x < width and 0 <= y < height:
                rgb_matrix[y][x] = (255, 0, 0)
    
    draw_line(px1, py1, px2, py2)
    
    return rgb_matrix

# Placeholder functions for rgb_matrix_to_bmp and save_bmp
# Replace these with your actual implementations

def rgb_matrix_to_bmp(rgb_matrix):
    height = len(rgb_matrix)
    width = len(rgb_matrix[0])
    padding = (4 - (width * 3) % 4) % 4
    file_size = 14 + 40 + (3 * width + padding) * height
    bmp = bytearray()
    bmp += b'BM'
    bmp += file_size.to_bytes(4, 'little')
    bmp += (0).to_bytes(4, 'little')
    bmp += (14 + 40).to_bytes(4, 'little')
    bmp += (40).to_bytes(4, 'little')
    bmp += width.to_bytes(4, 'little')
    bmp += height.to_bytes(4, 'little')
    bmp += (1).to_bytes(2, 'little')
    bmp += (24).to_bytes(2, 'little')
    bmp += (0).to_bytes(4, 'little')
    bmp += ((3 * width + padding) * height).to_bytes(4, 'little')
    bmp += (0).to_bytes(4, 'little') * 4
    for row in reversed(rgb_matrix):
        for (r, g, b) in row:
            bmp += bytes([b, g, r])
        bmp += b'\x00' * padding
    return bytes(bmp)

def save_bmp(filename, bmp_data):
    with open(filename, 'wb') as f:
        f.write(bmp_data)

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Train Linear Regression model
    theta0_initial = 0.0
    theta1_initial = 0.0
    learning_rate = 0.01
    num_iterations = 1000

    theta0_opt, theta1_opt, cost_history = gradient_descent(X, y, theta0_initial, theta1_initial, learning_rate, num_iterations)

    print(f"\nOptimized theta0 (Intercept): {theta0_opt:.4f}")
    print(f"Optimized theta1 (Slope): {theta1_opt:.4f}")

    # Plot and save the regression result
    rgb_matrix = plot_linear_regression(X, y, theta0_opt, theta1_opt, width=500, height=500, margin=50)
    bmp_data = rgb_matrix_to_bmp(rgb_matrix)
    save_bmp("./output/linear_regression.bmp", bmp_data)

    print("\nLinear regression plot saved to './output/linear_regression.bmp'")
