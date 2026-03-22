import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Geometric-Optics Phase Function for a 2D Hexagonal Ice Crystal
# ============================================================

# ------------------------------------------------------------
# 1. Physical parameters
# ------------------------------------------------------------
m_ice = 1.31
n_air = 1.0
n_ice = m_ice

x   = 500.0          # size parameter x = 2πa/λ
lam = 0.55e-6        # wavelength (m), e.g. 0.55 µm
a   = x*lam/(2*np.pi)

# ------------------------------------------------------------
# 2. Hexagon geometry via hex_vertices(a)
# ------------------------------------------------------------
def hex_vertices(a):
    """Regular hexagon of radius a, CCW order."""
    ang = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((a*np.cos(ang), a*np.sin(ang)))

def build_sides_from_vertices(verts):
    """Return list of sides as (p1, p2, n_out) using vertex list."""
    sides = []
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        e = p2 - p1
        t_hat = e / np.linalg.norm(e)
        # outward normal (clockwise)
        n_out = np.array([t_hat[1], -t_hat[0]])
        n_out /= np.linalg.norm(n_out)
        sides.append((p1, p2, n_out))
    return sides

def rotate(points, phi):
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    return points @ R.T

# ------------------------------------------------------------
# 3. Basic linear algebra helpers
# ------------------------------------------------------------
def intersect_ray_segment(ray_p, ray_v, p1, p2):
    s = p2 - p1
    ray_v = np.atleast_1d(ray_v).astype(float)
    s = np.atleast_1d(s).astype(float)
    A = np.column_stack((ray_v, -s))
    b = p1 - ray_p
    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        return None, None
    t, u = np.linalg.solve(A, b)
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t, u
    return None, None

def first_hit(pos, d, sides, eps=1e-10):
    """Find the nearest intersection of ray with any side.
    Returns (hit_point, normal) or None if no intersection."""
    best_t = np.inf
    best = None
    for p1, p2, n_out in sides:
        t, u = intersect_ray_segment(pos, d, p1, p2)
        if t is not None and t > eps and t < best_t:
            best_t = t
            best = (pos + t * d, n_out)
    return best

def reflect(v, n):
    return v - 2.0*np.dot(v, n)*n

# ------------------------------------------------------------
# 4. Fresnel coefficients and refraction
# ------------------------------------------------------------
def fresnel_sp(d, n, n1, n2):
    """Fresnel coefficients for s and p polarization (Marvin's method).
    Returns Rs, Rp, Ts, Tp (reflection and transmission coefficients).
    Args:
        d: incident ray direction
        n: surface normal
        n1: refractive index of incident medium
        n2: refractive index of transmitted medium"""
    cos_theta_i = np.clip(-np.dot(n, d), -1.0, 1.0)
    eta = n1 / n2
    st2 = eta * eta * max(0.0, 1.0 - cos_theta_i * cos_theta_i)
    if st2 > 1.0:
        return 1.0, 1.0, 0.0, 0.0
    ct = np.sqrt(max(0.0, 1.0 - st2))
    ds = n1 * cos_theta_i + n2 * ct
    dp = n2 * cos_theta_i + n1 * ct
    if abs(ds) < 1e-15 or abs(dp) < 1e-15:
        return 1.0, 1.0, 0.0, 0.0
    rs = (n1 * cos_theta_i - n2 * ct) / ds
    rp = (n2 * cos_theta_i - n1 * ct) / dp
    Rs = np.clip(rs * rs, 0.0, 1.0)
    Rp = np.clip(rp * rp, 0.0, 1.0)
    return Rs, Rp, 1.0 - Rs, 1.0 - Rp

def refract(v_in, n, n1, n2):
    """Refraction with s/p polarization coefficients (0 to 180 degrees).
    Returns (v_trans, tir, Rs, Rp, Ts, Tp)."""
    Rs, Rp, Ts, Tp = fresnel_sp(v_in, n, n1, n2)
    eta = n1 / n2
    ci = -np.dot(n, v_in)
    k = 1.0 - eta * eta * (1.0 - ci * ci)
    if k < 0.0:
        return None, True, 1.0, 1.0, 0.0, 0.0
    v_trans = eta*v_in + (eta*ci - np.sqrt(k))*n
    v_trans /= np.linalg.norm(v_trans)
    return v_trans, False, Rs, Rp, Ts, Tp

# ------------------------------------------------------------
# 5. Ray tracing through hexagon with splitting
# ------------------------------------------------------------
def trace_one_orientation(sides_rot, v_in, ray_start,
                          max_depth=7, weight_cut=1e-6):
    exit_rays = []
    # stack elements: (p, v, med, ws, wp, depth)
    # med: 0=air, 1=ice
    # ws, wp: s and p polarization weights
    stack = [(ray_start, v_in, 0, 0.5, 0.5, 0)]  # unpolarized: equal s/p
    while stack:
        p, v, med, ws, wp, depth = stack.pop()
        if (ws + wp) < weight_cut or depth > max_depth:
            continue

        n1 = n_air if med == 0 else n_ice
        n2 = n_ice if med == 0 else n_air

        # Find nearest intersection
        hit = first_hit(p, v, sides_rot)
        if hit is None:
            if med == 1:
                exit_rays.append((v, ws, wp))
            continue

        hit_point, n_out = hit

        # normal from medium 1 into medium 2
        if med == 0:
            n_int = -n_out  # air -> ice
        else:
            n_int = n_out   # ice -> air
        
        # Ensure normal points into the incident medium (standard ray tracing practice)
        if np.dot(n_int, v) > 0.0:
            n_int = -n_int

        v_trans, tir, Rs, Rp, Ts, Tp = refract(v, n_int, n1, n2)

        # reflected ray
        v_ref = reflect(v, n_int)
        ws_ref = ws * Rs
        wp_ref = wp * Rp
        if (ws_ref + wp_ref) > weight_cut:
            stack.append((hit_point, v_ref, med, ws_ref, wp_ref, depth+1))

        if tir:
            continue

        # transmitted ray
        ws_tr = ws * Ts
        wp_tr = wp * Tp
        if (ws_tr + wp_tr) <= weight_cut:
            continue

        if med == 0:
            # entering ice
            stack.append((hit_point, v_trans, 1, ws_tr, wp_tr, depth+1))
        else:
            # exiting to air
            exit_rays.append((v_trans, ws_tr, wp_tr))

    return exit_rays

# ------------------------------------------------------------
# 6. Batch ray tracing (processes all rays per orientation together)
# ------------------------------------------------------------
def trace_orientation_batch(verts, n_rays=200, max_depth=12, weight_cut=1e-6, eps=1e-9):
    """Trace all rays for a given orientation together."""
    y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
    if y1 <= y0:
        return []
    dy = (y1 - y0) / n_rays
    ys = y0 + (np.arange(n_rays) + 0.5) * dy
    x = np.min(verts[:, 0]) - 1e-6
    d0 = np.array([1.0, 0.0])
    I0 = 0.5 * dy
    
    sides = build_sides_from_vertices(verts)
    pool = [{"p": np.array([x, y]), "d": d0.copy(), "ws": I0, "wp": I0, "med": 0, "k": 0} for y in ys]
    out = []
    
    i = 0
    while i < len(pool):
        ray = pool[i]
        i += 1
        if (ray["ws"] + ray["wp"]) < weight_cut or ray["k"] > max_depth:
            continue
        
        n1 = n_air if ray["med"] == 0 else n_ice
        n2 = n_ice if ray["med"] == 0 else n_air
        
        hit = first_hit(ray["p"], ray["d"], sides, eps=1e-9)
        if hit is None:
            if ray["med"] == 1:
                out.append((ray["d"], ray["ws"], ray["wp"]))
            continue
        
        hit_point, n_out = hit
        
        if ray["med"] == 0:
            n_int = -n_out
        else:
            n_int = n_out
        if np.dot(n_int, ray["d"]) > 0.0:
            n_int = -n_int
        
        v_trans, tir, Rs, Rp, Ts, Tp = refract(ray["d"], n_int, n1, n2)
        
        v_ref = reflect(ray["d"], n_int)
        ws_ref = ray["ws"] * Rs
        wp_ref = ray["wp"] * Rp
        if (ws_ref + wp_ref) > weight_cut:
            pool.append({"p": hit_point + 1e-9 * v_ref, "d": v_ref, "ws": ws_ref, "wp": wp_ref, "med": ray["med"], "k": ray["k"] + 1})
        
        if tir:
            continue
        
        ws_tr = ray["ws"] * Ts
        wp_tr = ray["wp"] * Tp
        if (ws_tr + wp_tr) <= weight_cut:
            continue
        
        if ray["med"] == 0:
            pool.append({"p": hit_point + 1e-9 * v_trans, "d": v_trans, "ws": ws_tr, "wp": wp_tr, "med": 1, "k": ray["k"] + 1})
        else:
            out.append((v_trans, ws_tr, wp_tr))
    
    return out

# ------------------------------------------------------------
# 7. Setting variables: 
# ------------------------------------------------------------
theta_inc = 0.0              # 0 rad: along +x axis
v_in = np.array([np.cos(theta_inc), np.sin(theta_inc)])
v_in /= np.linalg.norm(v_in)

ray_start_base = -10.0*a*v_in

theta_min = 0.0
theta_max = 180.0
scatter_bin_deg = 1.0  # Increased from 2.0 for smoother output
num_bins  = int((theta_max - theta_min)/scatter_bin_deg) + 1
angles    = np.linspace(theta_min, theta_max, num_bins)
phase_counts_s = np.zeros_like(angles)  # s-polarization
phase_counts_p = np.zeros_like(angles)  # p-polarization
phase_counts = np.zeros_like(angles)    # total

orient_bin_deg = 1.0  # Hexagon orientation sampling: 1 degree (360 orientations)
thetas = np.arange(0.0, 360.0, orient_bin_deg)
n_rays = 1000  # Increased from 100

# base hexagon vertices
verts_base = hex_vertices(a)

for th in thetas:
    verts_rot = rotate(verts_base, np.deg2rad(th))
    sides_rot = build_sides_from_vertices(verts_rot)
    
    # Trace all rays for this orientation together
    y0, y1 = np.min(verts_rot[:, 1]), np.max(verts_rot[:, 1])
    if y1 > y0:
        dy = (y1 - y0) / n_rays
        ys = y0 + (np.arange(n_rays) + 0.5) * dy
        x = np.min(verts_rot[:, 0]) - 1e-6
        d0 = np.array([1.0, 0.0])
        I0 = 1.0 / n_rays  # Each ray has equal intensity, normalized by number of rays
        
        orient_count = 0
        for y in ys:
            ray_pos = np.array([x, y])
            exit_rays = trace_one_orientation(sides_rot, d0, ray_pos, max_depth=15, weight_cut=1e-6)
            
            if exit_rays:
                orient_count += len(exit_rays)
            
            for v_out, ws, wp in exit_rays:
                cos_theta = np.dot(d0, v_out)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.rad2deg(np.arccos(cos_theta))
                idx = np.argmin(np.abs(angles - theta))
                # Accumulate s and p components separately
                phase_counts_s[idx] += ws * I0 * 0.5  # 0.5 weight for s-polarization
                phase_counts_p[idx] += wp * I0 * 0.5  # 0.5 weight for p-polarization
                phase_counts[idx] += (ws + wp) * I0 * 0.5  # Total is average of s and p
        
        #print(f"Angle {th:.1f}°: {orient_count} exit rays")

# Normalize by number of orientations
if len(thetas) > 0:
    phase_counts_s /= len(thetas)
    phase_counts_p /= len(thetas)
    phase_counts /= len(thetas)

print(f"Total orientations: {len(thetas)}")

print(f"Total orientations: {len(thetas)}")
print(f"Phase counts (total) sum: {phase_counts.sum()}")
print(f"Phase counts (s-pol) sum: {phase_counts_s.sum()}")
print(f"Phase counts (p-pol) sum: {phase_counts_p.sum()}")

if phase_counts.sum() > 0:
    phase = (phase_counts_s + phase_counts_p) / (phase_counts_s.sum() + phase_counts_p.sum())
    phase_s = phase_counts_s / phase_counts_s.sum()
    phase_p = phase_counts_p / phase_counts_p.sum()
else:
    phase = phase_counts
    phase_s = phase_counts_s
    phase_p = phase_counts_p

# Calculate DoLP (Degree of Linear Polarization)
dolp = np.zeros_like(phase)
m = phase_counts > 0
dolp[m] = (phase_counts_s[m] - phase_counts_p[m]) / phase_counts[m]

# ------------------------------------------------------------
# 8. Diffraction contribution
# ------------------------------------------------------------
def diffraction_phase_function(angles_deg, x, P_ray):
    """Compute diffraction contribution using the formula:
    P_diff = (P_ray / (2π*x)) * (1+cos(θ)) * [∫ (x*cos(d)*sin(x*cos(d)*sin(θ))/(x*cos(d)*sin(θ))) dd]^2
    
    Integration limits: d from 0 to π/6 (30 degrees)
    
    Args:
        angles_deg: scattering angles in degrees
        x: size parameter (2πa/λ)
        P_ray: ray optics phase function (normalized)
    
    Returns:
        P_diff: diffraction phase function
    """
    from scipy.integrate import quad
    
    angles_rad = np.deg2rad(angles_deg)
    phase_diff = np.zeros_like(angles_rad)
    
    for i, theta in enumerate(angles_rad):
        if abs(np.sin(theta)) < 1e-10:  # Forward scattering
            phase_diff[i] = P_ray[i] * ((1 + np.cos(theta))**2) / (2 * np.pi * x)
            continue
        
        # Define integrand: (x*cos(d)*sin(x*cos(d)*sin(θ))) / (x*cos(d)*sin(θ))
        def integrand(d):
            sin_theta = np.sin(theta)
            cos_d = np.cos(d)
            arg = x * cos_d * sin_theta
            if abs(arg) < 1e-10:
                return 0.0
            numerator = x * cos_d * np.sin(arg)
            denominator = arg
            return numerator / denominator
        
        # Integrate from 0 to π/6
        try:
            integral_result, _ = quad(integrand, 0, np.pi/6, limit=100)
        except:
            integral_result = 0.0
        
        # Compute P_diff using ray optics phase function
        phase_diff[i] = (P_ray[i] / (2 * np.pi * x)) * (1 + np.cos(theta)) * integral_result**2
    
    return phase_diff

# Compute diffraction phase function using ray optics result
phase_diff = diffraction_phase_function(angles, x, phase)

# Normalize diffraction to maximum value
if np.max(phase_diff) > 0:
    phase_diff_norm = phase_diff / np.max(phase_diff)
else:
    phase_diff_norm = phase_diff

# Combine ray optics and diffraction: P_total = 0.5*P_ray + 0.5*P_diff
phase_combined = 0.5 * phase + 0.5 * phase_diff_norm

print(f"Ray optics phase sum: {phase.sum():.6f}")
print(f"Diffraction phase sum: {phase_diff_norm.sum():.6f}")
print(f"Combined phase sum: {phase_combined.sum():.6f}")

# ------------------------------------------------------------
# 7. Plot phase function and polarization
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Phase function (log scale)
ax = axes[0]
phase_log = np.copy(phase_combined)  # Use combined phase (ray + diffraction)
phase_log[phase_log <= 0] = 1e-12
ax.plot(angles, phase_log, color='#1f4788', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Scattering angle θ (degrees)')
ax.set_ylabel('P(θ) (log scale)')
ax.set_title('Phase Function (Ray + Diffraction)')
ax.grid(True, which='both')
ax.set_xlim(0, 180)

# DoLP (Degree of Linear Polarization)
ax = axes[1]
ax.plot(angles, dolp, 'g-', linewidth=2)
ax.set_xlabel('Scattering angle θ (degrees)')
ax.set_ylabel('DoLP')
ax.set_title('Degree of Linear Polarization')
ax.grid(True)
ax.set_xlim(0, 180)
ax.set_ylim(-1.0, 1.0)

plt.tight_layout()
import os
save_path = os.path.expanduser("~/hexagon_phase_function_polarized.png")
plt.savefig(save_path)
print(f"Figure saved to {save_path}")
plt.show()