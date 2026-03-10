import numpy as np
import matplotlib.pyplot as plt


def mie_wiscombe(m, x, nang=361):

    theta = np.linspace(0, np.pi, nang)
    mu = np.cos(theta)

    nmax = int(x + 4*x**(1/3) + 2)
    nmx = int(max(nmax, abs(m*x)) + 15)

    mx = m*x

    # --- Downward recurrence for logarithmic derivative D_n
    D = np.zeros(nmx+1, dtype=complex)

    for n in range(nmx, 0, -1):
        D[n-1] = (n/mx) - 1/(D[n] + n/mx)

    # --- Riccati–Bessel functions
    psi0 = np.cos(x)
    psi1 = np.sin(x)

    chi0 = -np.sin(x)
    chi1 = np.cos(x)

    xi1 = psi1 - 1j*chi1

    S1 = np.zeros(nang, dtype=complex)
    S2 = np.zeros(nang, dtype=complex)

    pi_nm1 = np.zeros(nang)
    pi_n = np.ones(nang)

    for n in range(1, nmax+1):

        psi = (2*n-1)/x*psi1 - psi0
        chi = (2*n-1)/x*chi1 - chi0
        xi = psi - 1j*chi

        an = ((D[n]/m + n/x)*psi - psi1) / ((D[n]/m + n/x)*xi - xi1)
        bn = ((m*D[n] + n/x)*psi - psi1) / ((m*D[n] + n/x)*xi - xi1)

        tau_n = n*mu*pi_n - (n+1)*pi_nm1
        coef = (2*n+1)/(n*(n+1))

        S1 += coef*(an*pi_n + bn*tau_n)
        S2 += coef*(an*tau_n + bn*pi_n)

        pi_np1 = ((2*n+1)/(n+1))*mu*pi_n - (n/(n+1))*pi_nm1

        pi_nm1 = pi_n
        pi_n = pi_np1

        psi0 = psi1
        psi1 = psi

        chi0 = chi1
        chi1 = chi

        xi1 = psi1 - 1j*chi1

    I = (np.abs(S1)**2 + np.abs(S2)**2)/2
    P = (np.abs(S2)**2 - np.abs(S1)**2)/(np.abs(S2)**2 + np.abs(S1)**2)

    return theta, I, P


# Parameters
m = 1.33
x = 500

theta, phase, pol = mie_wiscombe(m, x)

theta_deg = np.degrees(theta)

plt.figure()
plt.semilogy(theta_deg, phase)
plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Phase Function")
plt.title("Phase Function (Wiscombe Mie, m=1.33, x=500)")
plt.grid()

plt.figure()
plt.plot(theta_deg, pol)
plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Degree of Linear Polarization")
plt.title("Polarization (Wiscombe Mie, m=1.33, x=500)")
plt.grid()

# --- Polar plot of phase function ---
plt.figure()

ax = plt.subplot(111, projection='polar')

# use log scale so forward peak is visible
Iplot = np.log10(phase / np.max(phase) + 1e-20)

ax.plot(theta, Iplot)

ax.set_theta_zero_location("N")   # 0° at top
ax.set_theta_direction(-1)        # clockwise

ax.set_title("Polar Plot of Phase Function (log scale)")

plt.show()

