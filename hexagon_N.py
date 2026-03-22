import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#*This version is no where close to the correct phase function*
#  Geometric-Optics Phase Function for a 2D Hexagonal Ice Crystal
#  - Unpolarized incident light
#  - Refractive index of ice: m = 1.31
#  - Size parameter x = 2πa/λ = 500  (sets physical size a for a chosen λ)
#  - Random orientations of the hexagon
#  - Refraction, reflection, Fresnel coefficients
#  - Multiple internal interactions (ray splitting)
# ============================================================

# ------------------------------------------------------------
# 1. Physical parameters
# ------------------------------------------------------------
m_ice = 1.31
n_air = 1.0
n_ice = m_ice

x = 500.0                # size parameter x = 2πa/λ
lam = 0.55e-6            # wavelength (m), e.g. 0.55 µm
a   = x * lam / (2*np.pi)

# ------------------------------------------------------------
# 2. Geometry: regular hexagon of max dimension 2a
# ------------------------------------------------------------
# vertices with "flat top" hexagon, centered at origin
verts_base = np.array([
    [ a,    0.0],
    [ a/2,  np.sqrt(3)*a/2],
    [-a/2,  np.sqrt(3)*a/2],
    [-a,    0.0],
    [-a/2, -np.sqrt(3)*a/2],
    [ a/2, -np.sqrt(3)*a/2]
])

def build_sides(verts):
    """Return list of sides as (p1, p2, n_out) where n_out is outward unit normal."""
    sides = []
    for i in range(len(verts)):
        p1 = verts[i]
        p2 = verts[(i + 1) % len(verts)]
        e = p2 - p1
        t_hat = e / np.linalg.norm(e)
        # outward normal (choose clockwise orientation)
        n_out = np.array([t_hat[1], -t_hat[0]])
        n_out /= np.linalg.norm(n_out)
        sides.append((p1, p2, n_out))
    return sides

sides_base = build_sides(verts_base)

# ------------------------------------------------------------
# 3. Basic linear algebra helpers
# ------------------------------------------------------------
def rotate(points, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    return points @ R.T

def intersect_ray_segment(ray_p, ray_v, p1, p2):
    """
    Intersect ray r(t) = ray_p + t*ray_v, t>=0 with segment between p1 and p2.
    Returns (t, u) where p = p1 + u*(p2-p1); or (None,None) if no intersection.
    """
    s = p2 - p1
    A = np.column_stack((ray_v, -s))
    b = p1 - ray_p
    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        return None, None
    t, u = np.linalg.solve(A, b)
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t, u
    return None, None

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

# ------------------------------------------------------------
# 4. Fresnel coefficients and refraction
# ------------------------------------------------------------
def fresnel_unpolarized(cos_theta_i, n1, n2):
    """
    Fresnel reflection/transmission coefficients for unpolarized light.
    Returns R (power reflectance), T (power transmittance for intensity).
    cos_theta_i: cosine of incidence angle (>=0, between ray and surface normal)
    n1: index of incident medium
    n2: index of transmitted medium
    """
    # Handle near-normal incidence robustly
    cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)

    # Snell: n1 sin θi = n2 sin θt
    sin_theta_i2 = max(0.0, 1.0 - cos_theta_i**2)
    sin_theta_t2 = (n1/n2)**2 * sin_theta_i2
    if sin_theta_t2 > 1.0:  # total internal reflection
        return 1.0, 0.0

    cos_theta_t = np.sqrt(1.0 - sin_theta_t2)

    # Fresnel for s and p polarization
    rs = ((n1 * cos_theta_i - n2 * cos_theta_t) /
          (n1 * cos_theta_i + n2 * cos_theta_t))
    rp = ((n2 * cos_theta_i - n1 * cos_theta_t) /
          (n2 * cos_theta_i + n1 * cos_theta_t))

    Rs = rs**2
    Rp = rp**2
    R = 0.5 * (Rs + Rp)
    T = 1.0 - R   # no absorption; purely lossless

    return R, T

def refract(v_in, n, n1, n2):
    """
    Compute transmitted direction via vector Snell's law.
    v_in: incident unit vector (pointing toward the interface)
    n:    unit normal pointing from medium 1 to medium 2
    n1, n2: refractive indices
    Returns (v_trans, tir_flag, R, T)
      - v_trans: transmitted direction (None if TIR)
      - tir_flag: True if total internal reflection
      - R, T: Fresnel reflection and transmission coefficients (unpolarized)
    """
    # cos θi between incoming ray and normal (both pointing into interface from medium 1)
    cos_theta_i = np.dot(v_in, n)
    cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)

    R, T = fresnel_unpolarized(cos_theta_i, n1, n2)

    # Check for TIR
    sin_theta_i2 = max(0.0, 1.0 - cos_theta_i**2)
    eta = n1 / n2
    k = 1.0 - eta**2 * sin_theta_i2
    if k < 0.0:
        # total internal reflection
        return None, True, 1.0, 0.0

    # transmitted direction (standard vector form)
    v_trans = eta * v_in + (eta * cos_theta_i - np.sqrt(k)) * n
    v_trans /= np.linalg.norm(v_trans)

    return v_trans, False, R, T

# ------------------------------------------------------------
# 5. Ray tracing through hexagon with splitting
# ------------------------------------------------------------
def trace_one_orientation(sides_rot, v_in, ray_start, max_depth=6, weight_cut=1e-4):
    """
    Trace all rays for a single crystal orientation.
    Returns list of (exit_direction, weight) for all rays that leave to air.
    Uses a stack of rays; at each boundary we split into reflected and transmitted
    components according to Fresnel coefficients.
    """
    exit_rays = []
    # Each ray: (position, direction, medium_index, weight)
    # medium_index: 0 = air, 1 = ice
    stack = [(ray_start, v_in, 0, 1.0, 0)]  # (p, v, med, weight, depth)

    while stack:
        p, v, med, w, depth = stack.pop()
        if w < weight_cut or depth > max_depth:
            continue

        # Determine index of current medium and next
        n1 = n_air if med == 0 else n_ice
        n2 = n_ice if med == 0 else n_air

        # Find nearest intersection with any side
        best_t = None
        best_side = None
        for p1, p2, n_out in sides_rot:
            t, u = intersect_ray_segment(p, v, p1, p2)
            if t is not None and (best_t is None or t < best_t):
                best_t = t
                best_side = (p1, p2, n_out)
        if best_side is None:
            # Ray left without hitting crystal (if in air) or numerical issue
            if med == 1:
                # Shouldn't happen often, but treat as escaped
                exit_rays.append((v, w))
            continue

        p1, p2, n_out = best_side
        hit_point = p + best_t * v

        # Normal from medium 1 to medium 2:
        # if med==0 (air->ice), medium 1 is outside: n_out points outwards from ice,
        # but for enter we need normal pointing from air to ice -> use -n_out
        if med == 0:
            n_int = -n_out
        else:
            n_int = n_out

        # Refraction + Fresnel coefficients
        v_trans, tir, R, T = refract(v, n_int, n1, n2)

        # First, always have a reflected component (unless T=1 exactly)
        v_ref = reflect(v, n_int)
        w_ref = w * R
        if w_ref > weight_cut:
            # reflected ray stays in same medium
            stack.append((hit_point, v_ref, med, w_ref, depth + 1))

        if tir:
            # total internal reflection; no transmitted ray
            continue

        # Transmitted component
        w_tr = w * T
        if w_tr <= weight_cut:
            continue

        if med == 0:
            # entering ice
            stack.append((hit_point, v_trans, 1, w_tr, depth + 1))
        else:
            # exiting to air -> record outgoing ray
            exit_rays.append((v_trans, w_tr))

    return exit_rays

# ------------------------------------------------------------
# 6. Monte Carlo over orientations and build phase function
# ------------------------------------------------------------
# Incident direction (in air)
theta_inc = 0.0  # zenith incidence for symmetric halos
v_in = np.array([np.cos(theta_inc), np.sin(theta_inc)])
v_in /= np.linalg.norm(v_in)

# Place starting point 10 radii away along reverse direction
ray_start_base = -10.0 * a * v_in

# Scattering angle binning parameters
theta_min = 0.0      # degrees
theta_max = 180.0    # degrees
d_theta = 0.5       # bin width in degrees
num_bins = int((theta_max - theta_min) / d_theta) + 1
angles = np.linspace(theta_min, theta_max, num_bins)
phase_counts = np.zeros_like(angles, dtype=float)

N_orient = 5000
rng = np.random.default_rng(1)

total_exit_rays = 0

for _ in range(N_orient):
    # Randomly rotate the crystal
    phi = rng.uniform(0, 2*np.pi)
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    verts_rot = rotate(verts_base, phi)
    sides_rot = build_sides(verts_rot)

    exit_rays = trace_one_orientation(sides_rot, v_in, ray_start_base,
                                      max_depth=4, weight_cut=1e-6)

    total_exit_rays += len(exit_rays)

    # Convert each exit ray to scattering angle, accumulate weight
    for v_out, w in exit_rays:
        cos_theta = np.dot(v_in, v_out)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.rad2deg(np.arccos(cos_theta))
        idx = np.argmin(np.abs(angles - theta))
        phase_counts[idx] += w

print(f"Total exit rays: {total_exit_rays}")
print(f"Phase counts sum: {phase_counts.sum()}")

# Normalize phase function to unit integral (approx via sum)
if phase_counts.sum() > 0:
    phase = phase_counts / phase_counts.sum()
else:
    phase = phase_counts

# ------------------------------------------------------------
# 7. Plot phase function (log scale only)
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(8,5))

# log scale (avoid zeros)
phase_log = np.copy(phase)
phase_log[phase_log <= 0] = 1e-12
ax.plot(angles, phase_log, drawstyle='steps-mid')
ax.set_yscale('log')
ax.set_xlabel('Scattering angle θ (degrees)')
ax.set_ylabel('P(θ) (log scale)')
ax.set_title('Hexagonal ice crystal phase function\n(geometric optics, log scale)')
ax.grid(True, which='both')
ax.set_xlim(0, 180)

plt.tight_layout()
#plt.savefig('phase_function.png')
plt.show()