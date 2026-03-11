import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Geometric-optics scattering by a large sphere (2D slice model)
# ------------------------------------------------------------
# Assumptions:
#   1) Large sphere: geometric optics is appropriate (x = 500).
#   2) We illuminate a 2D slice of the sphere's projection:
#      rays are sampled uniformly in impact parameter b in [-1, 1].
#   3) Unpolarized incident light = equal s and p power.
#   4) We include:
#         - external reflection (p = 0)
#         - transmission with multiple internal reflections (p >= 1)
#   5) No diffraction, interference, tunneling, or surface-wave terms.
#
# Output:
#   - phase function P(theta) for the 2D slice model
#   - degree of linear polarization DoLP(theta)
#
# Notes:
#   - theta is the scattering angle in [0, pi] measured from
#     the forward direction.
#   - Positive DoLP means s-polarized dominates in this slice model.
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n

def reflect(k, n):
    """
    Reflect direction k about surface normal n.
    """
    k = normalize(k)
    n = normalize(n)
    return normalize(k - 2.0 * np.dot(k, n) * n)

def refract(k, n, n1, n2):
    """
    Refract a ray from medium n1 into medium n2.
    k  : incident unit direction
    n  : outward unit normal of the interface
    Returns transmitted unit direction, or None for total internal reflection.
    """
    k = normalize(k)
    n = normalize(n)

    # Make sure n points against the incident ray
    cos_i = -np.dot(n, k)
    if cos_i < 0:
        n = -n
        cos_i = -np.dot(n, k)

    eta = n1 / n2
    sin2_t = eta**2 * max(0.0, 1.0 - cos_i**2)

    if sin2_t > 1.0:
        return None  # total internal reflection

    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))
    t = eta * k + (eta * cos_i - cos_t) * n
    return normalize(t)

def fresnel_power_coefficients(n1, n2, cos_i):
    """
    Fresnel power coefficients for s and p polarizations.

    Returns:
        Rs, Rp, Ts, Tp

    where Rs/Rp are reflected power fractions,
    and Ts/Tp are transmitted power fractions.

    For lossless dielectrics: T = 1 - R.
    """
    cos_i = abs(np.clip(cos_i, -1.0, 1.0))
    sin2_i = max(0.0, 1.0 - cos_i**2)

    eta = n1 / n2
    sin2_t = eta**2 * sin2_i

    if sin2_t > 1.0:
        # total internal reflection
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
    Starting from a point P on/inside the sphere, march along direction k
    and find the next intersection with x^2 + z^2 = radius^2.
    """
    P = np.asarray(P, dtype=float)
    k = normalize(k)

    # Solve |P + t k|^2 = R^2
    B = 2.0 * np.dot(P, k)
    C = np.dot(P, P) - radius**2
    disc = B**2 - 4.0 * C

    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / 2.0
    t2 = (-B + sqrt_disc) / 2.0

    valid = [t for t in (t1, t2) if t > eps]
    if not valid:
        return None

    t = min(valid)
    return P + t * k

def trace_one_ray(b, m=1.33, p_max=8):
    """
    Trace one incident ray with impact parameter b through a unit sphere.

    Geometry:
      - 2D x-z plane
      - incident direction is +z
      - sphere centered at origin, radius = 1

    Returns a list of scattered contributions:
      [(theta, Is, Ip, order), ...]

    where
      theta = scattering angle in radians
      Is    = s-polarized power contribution
      Ip    = p-polarized power contribution
      order = ray order:
                0 = external reflection
                1 = directly transmitted ray (no internal reflection)
                2 = one internal reflection
                3 = two internal reflections
                ...
    """
    if abs(b) >= 1.0:
        return []

    contributions = []

    # Incident ray travels along +z
    k_in = np.array([0.0, 1.0])

    # Entry point on front hemisphere
    P1 = np.array([b, -np.sqrt(1.0 - b*b)])
    n1 = normalize(P1)   # outward normal

    # Fresnel coefficients at entry: air -> sphere
    cos_i1 = -np.dot(n1, k_in)
    Rs1, Rp1, Ts1, Tp1 = fresnel_power_coefficients(1.0, m, cos_i1)

    # Unpolarized incident light: half s, half p
    I0s = 0.5
    I0p = 0.5

    # ----- p = 0 : external reflection -----
    k_ref = reflect(k_in, n1)
    theta_ref = np.arccos(np.clip(np.dot(k_in, k_ref), -1.0, 1.0))
    contributions.append((theta_ref, I0s * Rs1, I0p * Rp1, 0))

    # ----- transmitted into sphere -----
    k_inside = refract(k_in, n1, 1.0, m)
    if k_inside is None:
        return contributions  # should not happen for air -> water

    Is = I0s * Ts1
    Ip = I0p * Tp1
    P = P1

    # Ray orders p >= 1
    for order in range(1, p_max + 1):
        # Find the next boundary hit
        P2 = next_sphere_intersection(P + 1e-10 * k_inside, k_inside, radius=1.0)
        if P2 is None:
            break

        n2 = normalize(P2)  # outward normal at the exit/reflection point
        cos_i2 = -np.dot(n2, k_inside)

        # Sphere -> air coefficients
        Rs2, Rp2, Ts2, Tp2 = fresnel_power_coefficients(m, 1.0, cos_i2)

        # Transmitted-out contribution for this order
        k_out = refract(k_inside, n2, m, 1.0)
        if k_out is not None:
            theta_out = np.arccos(np.clip(np.dot(k_in, k_out), -1.0, 1.0))
            contributions.append((theta_out, Is * Ts2, Ip * Tp2, order))

        # Continue the internally reflected branch
        Is *= Rs2
        Ip *= Rp2

        if (Is + Ip) < 1e-14:
            break

        k_inside = reflect(k_inside, n2)
        P = P2

    return contributions

def compute_phase_and_polarization(
    m=1.33,
    x=500.0,
    n_rays=200000,
    n_bins=720,
    p_max=8
):
    """
    Compute phase function and DoLP from many traced rays.

    Because the user requested a *slice* through the sphere projection,
    rays are sampled uniformly in b in [-1, 1].

    x is included as an input to document the large-particle regime.
    In pure geometric optics, the angular shape depends on m and the
    ray geometry, not explicitly on x.
    """
    # Uniform sampling across a 2D slice of the projected diameter
    b_vals = np.linspace(-0.999999, 0.999999, n_rays)
    db = b_vals[1] - b_vals[0]

    # Histogram in scattering angle theta in [0, pi]
    edges = np.linspace(0.0, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_s = np.zeros(n_bins)
    hist_p = np.zeros(n_bins)

    # Optional order-resolved storage
    order_hist = {}

    for b in b_vals:
        ray_contribs = trace_one_ray(b, m=m, p_max=p_max)

        for theta, Is, Ip, order in ray_contribs:
            idx = np.searchsorted(edges, theta, side="right") - 1
            if 0 <= idx < n_bins:
                # Weight by db since this is a uniform slice model
                hist_s[idx] += Is * db
                hist_p[idx] += Ip * db

                if order not in order_hist:
                    order_hist[order] = np.zeros(n_bins)
                order_hist[order][idx] += (Is + Ip) * db

    I_total = hist_s + hist_p

    # Slice-model phase function: normalize so integral over theta is 1
    dtheta = edges[1] - edges[0]
    norm = np.sum(I_total) * dtheta
    if norm > 0:
        phase = I_total / norm
        for order in order_hist:
            order_hist[order] /= norm
    else:
        phase = I_total

    # Degree of linear polarization
    dolp = np.zeros_like(I_total)
    mask = I_total > 0
    dolp[mask] = (hist_s[mask] - hist_p[mask]) / I_total[mask]

    return centers, phase, dolp, order_hist

def main():
    # User's parameters
    m = 1.33
    x = 500.0

    # Numerical settings
    n_rays = 120000   # increase for smoother curves
    n_bins = 720
    p_max = 8

    theta, phase, dolp, order_hist = compute_phase_and_polarization(
        m=m,
        x=x,
        n_rays=n_rays,
        n_bins=n_bins,
        p_max=p_max
    )

    theta_deg = np.degrees(theta)

    # Save data
    # data = np.column_stack([theta_deg, phase, dolp])
    # header = "theta_deg, phase_function_slice_model, degree_of_linear_polarization"
    # np.savetxt("sphere_go_phase_dolp.csv", data, delimiter=",", header=header, comments="")
    # print("Saved: sphere_go_phase_dolp.csv")

    # Plot phase function
    plt.figure(figsize=(8, 5))
    plt.semilogy(theta_deg, phase + 1e-30)
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Phase function P(theta)")
    plt.title(f"Geometric-optics phase function (slice model), m={m}, x={x:g}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot DoLP
    plt.figure(figsize=(8, 5))
    plt.plot(theta_deg, dolp)
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Degree of linear polarization")
    plt.title(f"DoLP (slice model), m={m}, x={x:g}")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Optional: order-resolved phase contributions
    plt.figure(figsize=(8, 5))
    for order in sorted(order_hist.keys()):
        if order <= 4:  # show first few orders clearly
            plt.semilogy(theta_deg, order_hist[order] + 1e-30, label=f"p={order}")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Order contribution")
    plt.title("Order-resolved contributions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()