import numpy as np
from collections import Counter

grades = [8, 9, 10, 7, 9]

mean = np.mean(grades)
median = np.median(grades)
mode = Counter(grades).most_common(1)[0][0]

print("Average:", mean)
print("Median:", median)
print("Mode:", mode)