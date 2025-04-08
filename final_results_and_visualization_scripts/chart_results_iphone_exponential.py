import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def weighted_std(values, weights):
    """Calculate the weighted standard deviation."""
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)

# Read total.csv
total_df = pd.read_csv('iphone_total_exp.csv', header=None)
total_values = total_df.iloc[:, 1].values  # Assuming the first column is the index

# Read deviation.csv
deviation_values = []
with open('iphone_deviation_exp.csv', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        index = parts[0]  # First value is the index
        pairs = parts[1:]  # Remaining values are data pairs
        values = []
        weights = []
        for pair in pairs:
            if ':' in pair:
                value, weight = map(float, pair.split(':'))
                values.append(value)
                weights.append(weight)
        if values and weights:
            deviation_values.append(weighted_std(np.array(values), np.array(weights)))
        else:
            deviation_values.append(0)  # Handle empty lines

# Ensure data integrity
if len(total_values) != len(deviation_values):
    raise ValueError("Mismatch in data length between total.csv and deviation.csv")

# Multiply deviation by total values
scaled_deviation = np.array(deviation_values) * total_values

# Normalize to percentage scale
max_value = max(total_values)
norm_total_values = (total_values / max_value) * 100
norm_deviation = (scaled_deviation / max_value) * 100

# Identify 1%, 0.1%, and 0.01% values
percent_values = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
percent_indices = {}

for p in percent_values:
    crossings = np.where(np.isclose(norm_total_values, p, atol=0.02))[0]  # Find closest x values
    percent_indices[p] = crossings
    if len(crossings) > 0:
        print(f'{p}% Level crosses at X values: {crossings.tolist()}')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(len(norm_total_values)), norm_total_values, label='Total Values (Normalized)', color='blue')
plt.fill_between(range(len(norm_total_values)), norm_total_values - norm_deviation, norm_total_values + norm_deviation, color='red', alpha=0.3, label='Deviation Range')
plt.fill_between(range(len(norm_total_values)), 0, norm_deviation * 10, color='green', alpha=0.3, label='Deviation * 10')

# Mark only the 1% level
if len(percent_indices[1]) > 0:
    plt.axhline(y=1, color='black', linestyle='--', label='1% Level')
    for idx in percent_indices[1]:
        plt.scatter(idx, 1, color='black', s=10)

plt.xlabel('Days since launch')
plt.ylabel('Percentage of maximum hits')
plt.title('Total Values with Weighted Standard Deviation (Normalized)')
plt.legend()
plt.show()
