import csv
import numpy as np
import matplotlib.pyplot as plt

def read_accuracy(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        accuracy = [float(row[1]) for row in reader]
    return accuracy

def compute_moving_average(values, window_size):
    cumulative_sum = np.cumsum(values, dtype=float)
    cumulative_sum[window_size:] = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    moving_average = cumulative_sum[window_size - 1:] / window_size
    return moving_average

# Read accuracy data
shape_accuracy = read_accuracy('./mlruns/SDIM/run1/metrics/shape_accuracy')
scale_accuracy = read_accuracy('./mlruns/SDIM/run1/metrics/scale_accuracy')
orientation_accuracy = read_accuracy('./mlruns/SDIM/run1/metrics/orientation_accuracy')
x_floor_hue_accuracy = read_accuracy('./mlruns/EDIM/run1/metrics/x_floor_hue_accuracy')
x_wall_hue_accuracy = read_accuracy('./mlruns/SDIM/run1/metrics/x_wall_hue_accuracy')
x_object_hue_accuracy = read_accuracy('./mlruns/SDIM/run1/metrics/x_object_hue_accuracy')

# Set the window size for moving average
window_size = 100  # You can adjust this value as needed

# Compute moving averages
shape_moving_average = compute_moving_average(shape_accuracy, window_size)
scale_moving_average = compute_moving_average(scale_accuracy, window_size)
orientation_moving_average = compute_moving_average(orientation_accuracy, window_size)
x_floor_hue_moving_average = compute_moving_average(x_floor_hue_accuracy, window_size)
x_wall_hue_moving_average = compute_moving_average(x_wall_hue_accuracy, window_size)
x_object_hue_moving_average = compute_moving_average(x_object_hue_accuracy, window_size)

# Find stable regions and means
metrics = {
    "Shape Accuracy": shape_moving_average,
    "Scale Accuracy": scale_moving_average,
    "Orientation Accuracy": orientation_moving_average,
    "Floor Hue Accuracy": x_floor_hue_moving_average,
    "Wall Hue Accuracy": x_wall_hue_moving_average,
    "Object Hue Accuracy": x_object_hue_moving_average
}

means = {
    "Shape Accuracy": 0,
    "Scale Accuracy": 0,
    "Orientation Accuracy": 0,
    "Floor Hue Accuracy": 0,
    "Wall Hue Accuracy": 0,
    "Object Hue Accuracy": 0
}

fig, ax = plt.subplots(2, 3, figsize=(16, 6))


for idx, (title, moving_average) in enumerate(metrics.items()):
    grad = abs(np.gradient(moving_average))
    means[title] = np.mean(moving_average[np.argwhere(grad < 1e-12)])
    i, j = divmod(idx, 3)
    ax[i, j].plot(moving_average, label='$S_X$')
    ax[i, j].set_title(title)
    ax[i, j].set_ylim(0, 1.1)
    ax[i, j].grid()
    ax[i, j].legend()
    

print(means)
plt.tight_layout()
plt.show()

