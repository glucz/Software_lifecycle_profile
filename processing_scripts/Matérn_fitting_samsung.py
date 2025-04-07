import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, WhiteKernel, RBF, Matern, ExpSineSquared

# Function to fit Gaussian Process Regression separately for the beginning and end parts
def fit_and_extrapolate(file_path):
    # Load data from file
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    
    # Apply a low-pass filter to smooth out noise and find the maximum region
    y_smooth = savgol_filter(y, window_length=11, polyorder=3)  # Increased smoothing window
    max_index = np.argmax(y_smooth)
    max_value = x[max_index]
    
    # Skip file if the first value is the maximum
    if max_index == 0:
        print(f"Skipping {file_path} as the first value is the maximum.")
        return None, None, None
    
    # Split data into beginning (up to max) and end (from max)
    x_start, y_start = x[:max_index+1], y[:max_index+1]
    x_end, y_end = x[max_index+1:], y[max_index+1:]
    
    # Fit Gaussian Process for the start 
    kernel_start = Matern(length_scale=5, nu=2.5) 
    gpr_start = GaussianProcessRegressor(kernel=kernel_start, alpha=2, normalize_y=True, n_restarts_optimizer=10)
    gpr_start.fit(x_start, y_start)
    y_start_pred, _ = gpr_start.predict(x_start, return_std=True)
    
    # Fit Gaussian Process for the end (capture long constant values and enforce gradual residual disappearance)
    kernel_end = Matern(length_scale=10, nu=1.5)
    gpr_end = GaussianProcessRegressor(kernel=kernel_end, alpha=0.65, normalize_y=True, n_restarts_optimizer=10)
    gpr_end.fit(x_end, y_end)
    y_end_pred, _ = gpr_end.predict(x_end, return_std=True)
    
    # Scale predictions within the range [0, max(y)] to ensure gradual decay
    y_start_pred = np.interp(y_start_pred, (y_start_pred.min(), y_start_pred.max()), (y.min(), y.max()))
    y_end_pred = np.interp(y_end_pred, (y_end_pred.min(), y_end_pred.max()), (0, y.max()))
    
    # Apply a smoothing function to ensure gradual decay
    decay_factor = np.linspace(1, 0, len(y_end_pred)) ** 2  # Quadratic decay
    y_end_pred *= decay_factor

    return x.flatten(), y, x_start.flatten(), y_start_pred, x_end.flatten(), y_end_pred

# Process all CSV files in a folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            result = fit_and_extrapolate(file_path)
            
            if result[0] is None:
                continue

            x_orig, y_orig, x_start, y_start_pred, x_end, y_end_pred = result
            y_fit = np.concatenate((y_start_pred, y_end_pred))
            
            # Save processed CSV
            output_csv = os.path.join(output_folder, f"processed_{file_name}")
            np.savetxt(output_csv, np.column_stack((x_orig, y_fit)), delimiter=',', header='X,Y', comments='')
            
            # Plot results
            plt.figure(figsize=(10, 5))
            plt.scatter(x_orig, y_orig, color='red', label='Original Data')
            plt.plot(x_start, y_start_pred, color='green', label='GPR expansion phase', linestyle='--')
            plt.plot(x_end, y_end_pred, color='blue', label='GPR decay phase', linestyle='--')
            plt.xlabel('Day index since software launch')
            plt.ylabel('Browser traffic volume')
            chart_title = file_name.replace('.csv', '')
            plt.title(f'Gaussian Process Regression Fit - {chart_title}')
            plt.legend()
            
            # Save graph
            output_img = os.path.join(output_folder, f"graph_{chart_title}.png")
            plt.savefig(output_img, dpi=300)
            plt.close()

# Example usage
input_folder = 'input_data_samsung'  # Replace with actual input folder path
output_folder = 'output_data_samsung_decay'  # Replace with actual output folder path
process_folder(input_folder, output_folder)
