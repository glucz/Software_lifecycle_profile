import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import warnings

# Exponential and logarithmic decay models
def exp_decay(x, a, b):
    x_clipped = np.clip(x, 0, 700)
    if np.any(x != x_clipped):
        warnings.warn("Clipping large x values in exp_decay to avoid overflow.")
    return a * np.exp(-b * x_clipped)

def log_decay(x, a, b):
    return a / (1 + b * x)

def enforce_monotonic_increase(y):
    y_fixed = np.copy(y)
    max_so_far = y_fixed[0]
    for i in range(1, len(y_fixed)):
        if y_fixed[i] < max_so_far:
            y_fixed[i] = max_so_far
        else:
            max_so_far = y_fixed[i]
    return y_fixed

def fit_and_extrapolate(file_path, use_log_decay=False):
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]

    y_smooth = savgol_filter(y, window_length=11, polyorder=3)
    max_index = np.argmax(y_smooth)

    if max_index == 0:
        print(f"Skipping {file_path} as the first value is the maximum.")
        return None

    x_start, y_start = x[:max_index], y[:max_index]
    x_end, y_end = x[max_index:], y[max_index:]
    y_end_monotonic = enforce_monotonic_increase(y_end[::-1])[::-1]

    gpr_start = GaussianProcessRegressor(kernel=Matern(length_scale=5, nu=2.5), alpha=2, normalize_y=True)
    gpr_start.fit(x_start.reshape(-1, 1), y_start)
    y_start_pred = gpr_start.predict(x_start.reshape(-1, 1))

    try:
        shifted_x_end = x_end - x_end[0]
        max_y = y[max_index]
        b_guess = 1 / (shifted_x_end[-1] - shifted_x_end[0] + 1e-6)
        if use_log_decay:
            popt_end, _ = curve_fit(log_decay, shifted_x_end, y_end_monotonic, p0=(max_y, b_guess), maxfev=10000)
            y_end_pred = log_decay(shifted_x_end, *popt_end)
        else:
            popt_end, _ = curve_fit(exp_decay, shifted_x_end, y_end_monotonic, p0=(max_y, b_guess), maxfev=10000)
            y_end_pred = exp_decay(shifted_x_end, *popt_end)
        r2_end = r2_score(y_end_monotonic, y_end_pred)
        mse_end = mean_squared_error(y_end_monotonic, y_end_pred)
    except Exception as e:
        print(f"End segment fit failed for {file_path}: {e}")
        return None

    return x, y, x_start, y_start_pred, x_end, y_end_pred, y_start, y_end_monotonic, {
        "r2_end": r2_end, "mse_end": mse_end, "popt_end": popt_end
    }

def process_folder(input_folder, output_folder, use_log_decay=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    param_log = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            result = fit_and_extrapolate(file_path, use_log_decay=use_log_decay)

            if result is None:
                continue

            x_orig, y_orig, x_start, y_start_pred, x_end, y_end_pred, y_start_raw, y_end_monotonic, fit_params = result
            y_fit = np.concatenate((y_start_pred, y_end_pred))

            output_csv = os.path.join(output_folder, f"processed_{file_name}")
            np.savetxt(output_csv, np.column_stack((x_orig, y_fit)), delimiter=',', header='X,Y', comments='')

            chart_title = file_name.replace('.csv', '')
            param_log.append([
                chart_title,
                "NA", "NA", "NA", "NA",
                *fit_params["popt_end"], fit_params["r2_end"], fit_params["mse_end"]
            ])

            plt.figure(figsize=(10, 5))
            plt.scatter(x_orig, y_orig, color='red', label='Original Data')
            plt.plot(x_end, y_end_monotonic, color='purple', linestyle=':', label='Preprocessed Decay Segment')
            plt.plot(x_start, y_start_pred, color='green', label='GPR Expansion Phase', linestyle='--')
            decay_label = 'Log Decay Fit' if use_log_decay else 'Exponential Decay Fit'
            plt.plot(x_end, y_end_pred, color='blue', label=decay_label, linestyle='--')
            plt.plot(x_orig, y_fit, color='black', alpha=0.3, label='Combined Fit')
            plt.xlabel('Day index since software launch')
            plt.ylabel('Browser traffic volume')
            plt.title(f'{decay_label} - {chart_title}')
            plt.legend()

            output_img = os.path.join(output_folder, f"graph_{chart_title}.png")
            plt.savefig(output_img, dpi=300)
            plt.close()

    param_log_path = os.path.join(output_folder, "fit_parameters_summary.csv")
    with open(param_log_path, 'w') as f:
        f.write("File,L_start,k_start,R2_start,MSE_start,a_end,b_end,R2_end,MSE_end\n")
        for row in param_log:
            f.write(",".join(map(str, row)) + "\n")

input_folder = 'input_data_samsung'
process_folder(input_folder, 'output_data_samsung_exp', use_log_decay=False)
process_folder(input_folder, 'output_data_samsung_log', use_log_decay=True)
