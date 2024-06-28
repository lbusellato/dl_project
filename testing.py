import csv
import matplotlib.pyplot as plt

with open('shape_accuracy') as f:
    reader = csv.DictReader(f, delimiter=' ')
    curve = [row['a'] for row in reader]

plt.plot(curve)
plt.show()