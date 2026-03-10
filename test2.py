import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


def mie_phase_pol(m, x, nang=361):
    """
    Compute phase function and degree of linear polarization
    for unpolarized light scattered by a sphere.

    Parameters
    ----------
    m : float
        Refractive index
    x : float
        Size parameter
    nang : int
        Number of scattering angles

    Returns
    -------
    theta : array
        Scattering angles (radians)
    I : array
        Phase function
    P : array
        Degree of linear polarization
    """

    theta = np.linspace(0, np.pi, nang)
    mu = np.cos(theta)

    # Series truncation
    nmax = int(np.round(x + 4*x**(1/3) + 2))
    n = np.arange(1, nmax+1)

    mx = m*x

    # Bessel functions at x
    jn_x = spherical_jn(n, x)
    jn_x_p = spherical_jn(n, x, derivative=True)

    yn_x = spherical_yn(n, x)
    yn_x_p = spherical_yn(n, x, derivative=True)

    psi = x * jn_x
    psi_p = jn_x + x * jn_x_p

    chi = -x * yn_x
    chi_p = -(yn_x + x * yn_x_p)

    xi = psi + 1j*chi
    xi_p = psi_p + 1j*chi_p

    # Bessel functions at mx
    jn_mx = spherical_jn(n, mx)
    jn_mx_p = spherical_jn(n, mx, derivative=True)

    psi_m = mx * jn_mx
    psi_m_p = jn_mx + mx * jn_mx_p

    # Mie coefficients
    a = (m*psi_m*psi_p - psi*psi_m_p) / (m*psi_m*xi_p - xi*psi_m_p)
    b = (psi_m*psi_p - m*psi*psi_m_p) / (psi_m*xi_p - m*xi*psi_m_p)

    # Amplitude functions
    S1 = np.zeros_like(theta, dtype=complex)
    S2 = np.zeros_like(theta, dtype=complex)

    pi_nm1 = np.zeros_like(theta)
    pi_n = np.ones_like(theta)

    for ni in range(1, nmax+1):

        tau_n = ni*mu*pi_n - (ni+1)*pi_nm1
        coef = (2*ni+1)/(ni*(ni+1))

        S1 += coef*(a[ni-1]*pi_n + b[ni-1]*tau_n)
        S2 += coef*(a[ni-1]*tau_n + b[ni-1]*pi_n)

        pi_np1 = ((2*ni+1)/(ni+1))*mu*pi_n - (ni/(ni+1))*pi_nm1
        pi_nm1 = pi_n
        pi_n = pi_np1

    # Phase function
    I = (np.abs(S1)**2 + np.abs(S2)**2)/2

    # Degree of linear polarization
    P = (np.abs(S2)**2 - np.abs(S1)**2) / (np.abs(S2)**2 + np.abs(S1)**2)

    return theta, I, P


# Parameters
m = 1.33
x = 500

theta, phase, pol = mie_phase_pol(m, x)

# Convert to degrees for plotting
theta_deg = np.degrees(theta)

# Plot phase function
plt.figure()
plt.semilogy(theta_deg, phase)
plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Phase Function")
plt.title("Mie Phase Function (m=1.33, x=500)")
plt.grid()

# Plot polarization
plt.figure()
plt.plot(theta_deg, pol)
plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Degree of Linear Polarization")
plt.title("Polarization (m=1.33, x=500)")
plt.grid()

plt.show()