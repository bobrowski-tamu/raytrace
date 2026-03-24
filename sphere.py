import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1 as bessel_j1

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def reflect(k, n):
    k = normalize(k)
    n = normalize(n)
    return normalize(k - 2.0 * np.dot(k, n) * n)

def refract(k, n, n1, n2):
    """
    Refract ray k through interface with outward normal n.
    Returns transmitted unit vector, or None for total internal reflection.
    """
    k = normalize(k)
    n = normalize(n)

    cos_i = -np.dot(n, k)
    if cos_i < 0:
        n = -n
        cos_i = -np.dot(n, k)

    eta = n1 / n2
    sin2_t = eta**2 * max(0.0, 1.0 - cos_i**2)

    if sin2_t > 1.0:
        return None

    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))
    t = eta * k + (eta * cos_i - cos_t) * n
    return normalize(t)

def fresnel_coefficients(n1, n2, cos_i):
    """
    Fresnel coefficients for s and p polarization.
    Returns Rs, Rp, Ts, Tp
    """
    cos_i = abs(np.clip(cos_i, -1.0, 1.0))
    sin2_i = max(0.0, 1.0 - cos_i**2)

    eta = n1 / n2
    sin2_t = eta**2 * sin2_i

    if sin2_t > 1.0:
        return 1.0, 1.0, 0.0, 0.0

    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))

    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)

    Rs = rs**2
    Rp = rp**2
    Ts = 1.0 - Rs
    Tp = 1.0 - Rp

    return Rs, Rp, Ts, Tp

def next_sphere_intersection(P, k, radius=1.0, eps=1e-10):
    """
    From point P, move along direction k to next intersection with sphere.
    """
    P = np.asarray(P, dtype=float)
    k = normalize(k)

    B = 2.0 * np.dot(P, k)
    C = np.dot(P, P) - radius**2
    disc = B**2 - 4.0 * C

    if disc < 0:
        return None

    sq = np.sqrt(disc)
    t1 = (-B - sq) / 2.0
    t2 = (-B + sq) / 2.0

    valid = [t for t in (t1, t2) if t > eps]
    if not valid:
        return None

    t = min(valid)
    return P + t * k

def trace_one_ray(b, m=1.33, p_max=8):
    """
    Trace one ray with impact parameter b (0 <= b < 1).
    Returns [(theta, Is, Ip), ...]
    """
    if not (0.0 <= b < 1.0):
        return []

    contributions = []

    # Incident direction: +z in 2D (x,z) plane
    k_in = np.array([0.0, 1.0])

    # Entry point on front hemisphere
    P1 = np.array([b, -np.sqrt(1.0 - b*b)])
    n1 = normalize(P1)

    # Entry Fresnel coefficients: air -> sphere
    cos_i1 = -np.dot(n1, k_in)
    Rs1, Rp1, Ts1, Tp1 = fresnel_coefficients(1.0, m, cos_i1)

    # Unpolarized incident light
    I0s = 0.5
    I0p = 0.5

    # p = 0 external reflection
    k_ref = reflect(k_in, n1)
    theta_ref = np.arccos(np.clip(np.dot(k_in, k_ref), -1.0, 1.0))
    contributions.append((theta_ref, I0s * Rs1, I0p * Rp1))

    # Refracted ray into sphere
    k_inside = refract(k_in, n1, 1.0, m)
    if k_inside is None:
        return contributions

    Is = I0s * Ts1
    Ip = I0p * Tp1
    P = P1

    # p >= 1
    for _ in range(1, p_max + 1):
        P2 = next_sphere_intersection(P + 1e-10 * k_inside, k_inside)
        if P2 is None:
            break

        n2 = normalize(P2)
        cos_i2 = -np.dot(n2, k_inside)

        # sphere -> air
        Rs2, Rp2, Ts2, Tp2 = fresnel_coefficients(m, 1.0, cos_i2)

        # transmitted-out contribution
        k_out = refract(k_inside, n2, m, 1.0)
        if k_out is not None:
            theta_out = np.arccos(np.clip(np.dot(k_in, k_out), -1.0, 1.0))
            contributions.append((theta_out, Is * Ts2, Ip * Tp2))

        # internally reflected branch continues
        Is *= Rs2
        Ip *= Rp2

        if (Is + Ip) < 1e-14:
            break

        k_inside = reflect(k_inside, n2)
        P = P2

    return contributions

def compute_diffraction_phase(theta, x=500.0, n_bins=720):
    """
    Compute diffractrion
    """
    # Handle theta = 0 specially (avoid division by zero)
    phase_diff = np.zeros_like(theta)
    
    # Compute only for non-zero sin(theta)
    mask = np.sin(theta) > 1e-12
    sin_theta = np.sin(theta[mask])
    arg = x * sin_theta
    
    # Compute diffraction term
    bessel_term = 2.0 * bessel_j1(arg) / arg
    phase_diff[mask] = (bessel_term**2) * (1.0 + np.cos(theta[mask]))**2
    
    # Normalize: integral P(theta) dOmega = 1
    dtheta = theta[1] - theta[0]
    norm = np.sum(phase_diff * 2.0 * np.pi * np.sin(theta) * dtheta)
    if norm > 0:
        phase_diff /= norm
    
    return phase_diff

def compute_phase_and_dolp(m=1.33, n_rays=50000, n_bins=720, p_max=8):
    """
    Computes phase function and DoLP for a sphere.
    """
    # Sample impact parameter over projected disk radius
    b_vals = np.linspace(0.0, 0.999999, n_rays)
    db = b_vals[1] - b_vals[0]

    # Angular bins in [0, pi]
    edges = np.linspace(0.0, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dtheta = edges[1] - edges[0]

    hist_s = np.zeros(n_bins)
    hist_p = np.zeros(n_bins)

    for b in b_vals:
        # 3D sphere weighting: annulus on projected disk
        w = 2.0 * np.pi * b * db

        contribs = trace_one_ray(b, m=m, p_max=p_max)
        for theta, Is, Ip in contribs:
            idx = np.searchsorted(edges, theta, side="right") - 1
            if 0 <= idx < n_bins:
                hist_s[idx] += Is * w
                hist_p[idx] += Ip * w

    I_theta = hist_s + hist_p

    # Convert histogram in theta into phase function per unit solid angle
    phase = np.zeros_like(I_theta)
    mask = np.sin(centers) > 1e-12
    phase[mask] = I_theta[mask] / (2.0 * np.pi * np.sin(centers[mask]) * dtheta)

    # Normalize: integral P(theta) dOmega = 1
    norm = np.sum(phase * 2.0 * np.pi * np.sin(centers) * dtheta)
    if norm > 0:
        phase /= norm

    # Degree of linear polarization
    dolp = np.zeros_like(I_theta)
    mask2 = I_theta > 0
    dolp[mask2] = (hist_s[mask2] - hist_p[mask2]) / I_theta[mask2]

    return centers, phase, dolp

def combine_phase_functions(phase_geo, phase_diff):
    """
    Combines geometric optics and diffraction phase functions.
    """
    phase_combined = 0.5 * phase_geo + 0.5 * phase_diff
    return phase_combined

theta, phase, dolp = compute_phase_and_dolp(
    m=1.33,
    n_rays=50000,
    n_bins=720,
    p_max=15
)

# Compute diffraction phase function
x = 500.0
phase_diff = compute_diffraction_phase(theta, x=x, n_bins=720)

# Combine phase functions: 50% geometric optics + 50% diffraction
phase_combined = combine_phase_functions(phase, phase_diff)

theta_deg = np.degrees(theta)

# Plot DoLP
plt.figure(figsize=(8, 5))
plt.plot(theta_deg, dolp)
plt.xlabel("Scattering angle (deg)")
plt.ylabel("Degree of linear polarization (DOLP)")
plt.title(f"DoLP for sphere (geometric optics), m=1.33, x={x:.0f}")
plt.ylim(-1.05, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sphere_DOLP.png', dpi=150)

# Plot geometric optics phase alone
plt.figure(figsize=(8, 5))
plt.semilogy(theta_deg, phase + 1e-30)
plt.xlabel("Scattering angle (deg)")
plt.ylabel("Phase function P($\Theta$)")
plt.title(f"Sphere phase function (geometric optics), m=1.33, x={x:.0f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sphere_phase_function.png', dpi=150)

plt.show()
