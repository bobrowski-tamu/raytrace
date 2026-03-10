import numpy as np
import miepython
import matplotlib.pyplot as plt

# Input parameters
m = 1.33          # Refractive index (real)
X = 500           # Size parameter (X = 2pi * R / lambda)

# Scattering angles from 0 to 180 degrees
theta_deg = np.linspace(0, 180, 1000)
theta_rad = np.deg2rad(theta_deg)

# Compute S1 and S2 (scattering amplitudes)
S1, S2 = miepython.S1_S2(m, X, theta_rad)

# Phase function (normalized intensity for unpolarized light)
phase_function = (np.abs(S1)**2 + np.abs(S2)**2) / 2

# DOLP (degree of linear polarization)
DOLP = (np.abs(S1)**2 - np.abs(S2)**2) / (np.abs(S1)**2 + np.abs(S2)**2)

# Plot results
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.semilogy(theta_deg, phase_function)
plt.xlabel("Scattering Angle (deg)")
plt.ylabel("Phase Function (unnormalized)")
plt.title("Phase Function (X=500, m=1.33)")

plt.subplot(122)
plt.plot(theta_deg, DOLP)
plt.xlabel("Scattering Angle (deg)")
plt.ylabel("Degree of Linear Polarization")
plt.title("DOLP (X=500, m=1.33)")

plt.tight_layout()
plt.show()