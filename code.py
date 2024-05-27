import numpy as np
import matplotlib.pyplot as plt
import math
# Given matrices

K = np.array([[800, -600,0], [-600,1200,-600], [0,-600, 1400]])
M = np.array([[1,0,0], [0,2,0], [0,0,1]])
C = 0.1*M

# Frequency range
omega = np.linspace(0, 50, 10000)
H_22 = []
H_12 = []

# Calculate FRF
for w in omega:
    H = np.linalg.inv(K - w**2 * M + 1j * w * C)
    H_22.append(H[1, 1])
    H_12.append(H[0, 1])

H_22 = np.array(H_22)
H_12 = np.array(H_12)


import scipy.signal

magnitude_H22 = np.abs(H_22)

peak_indices, _ = scipy.signal.find_peaks(magnitude_H22, height=0)  
print("peak_indices:", peak_indices)
natural_frequencies = omega[peak_indices] 
peak_magnitudes= np.abs(H_22)[peak_indices]  

# 打印峰值
print("natural_frequencies:", natural_frequencies)
print("peak_magnitudes:", peak_magnitudes)


damping_ratios = []

for i, omega_n in enumerate(natural_frequencies):
    half_power_point = peak_magnitudes[i] / np.sqrt(2)
    # Find the closest frequencies to the half power points
    indices_left = np.argwhere(magnitude_H22[:peak_indices[i]] <= half_power_point)
    indices_right = np.argwhere(magnitude_H22[peak_indices[i]:] <= half_power_point) + peak_indices[i]
    
    if indices_left.size > 0 and indices_right.size > 0:
        omega_1 = omega[indices_left[-1]].flatten()[0]
        omega_2 = omega[indices_right[0]].flatten()[0]
        delta_omega = omega_2 - omega_1
        zeta = delta_omega / (2 * omega_n)
        damping_ratios.append(zeta)
    else:
        damping_ratios.append(np.nan)  # If we can't find half power points, assign NaN

for i in range(len(natural_frequencies)):
    print(f"Natural Frequency {i+1}: {natural_frequencies[i]:.4f} rad/s, Damping Ratio: {damping_ratios[i]:.6f}")