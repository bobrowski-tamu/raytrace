import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

m_ice = 1.31
x = 500.0                    # Size parameter
lam = 0.55e-6               # Wavelength (m)
a = x * lam / (2 * np.pi)   # Hexagon radius

n_rays = 1000             # Rays per orientation
n_orientations = 360       # Number of orientations (0-360°)
max_depth = 12              # Max internal reflections
weight_cut = 1e-8         # Weight cutoff threshold

scatter_bin_deg = 1.0     # Scattering angle bin width
n_bins = int(round(180.0 / scatter_bin_deg))


def hexagon_vertices(a=1.0, rotation=0.0):
    """Create regular hexagon vertices."""
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + rotation
    return np.stack((a*np.cos(angles), a*np.sin(angles)), axis=1)

def rotate_pts(pts, th):
    """Rotate points by angle th."""
    c, s = np.cos(th), np.sin(th)
    return pts @ np.array([[c, -s], [s, c]]).T

def norm(v):
    """Normalize vector."""
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n

def scatter_deg(v, inc=np.array([1.0, 0.0])):
    """Compute scattering angle in degrees."""
    c = np.clip(np.dot(norm(v), norm(inc)), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def intersect_ray_segment(ray_p, ray_v, p1, p2):
    """Find ray-segment intersection (simplified)."""
    v1 = ray_p - p1
    v2 = p2 - p1
    v3 = np.array([-ray_v[1], ray_v[0]])
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-10:
        return None, None
    t1 = (v2[0]*v1[1] - v2[1]*v1[0]) / dot
    t2 = np.dot(v1, v3) / dot
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return t1, t2
    return None, None

def first_hit(pos, d, verts):
    """Find nearest intersection with hexagon."""
    best_t = np.inf
    best_n = None
    for i in range(len(verts)):
        p1, p2 = verts[i], verts[(i + 1) % len(verts)]
        t, u = intersect_ray_segment(pos, d, p1, p2)
        if t is not None and t > 1e-10 and t < best_t:
            best_t = t
            e = p2 - p1
            n_out = norm(np.array([e[1], -e[0]]))
            best_n = (pos + t * d, n_out)
    return best_n

def fresnel_sp(d, n, n1, n2):
    """Fresnel coefficients for s and p polarization."""
    ci = np.clip(-np.dot(n, d), -1.0, 1.0)
    η = n1 / n2
    st2 = η * η * max(0.0, 1.0 - ci * ci)
    if st2 > 1.0:
        return 1.0, 1.0, 0.0, 0.0
    ct = np.sqrt(max(0.0, 1.0 - st2))
    ds = n1 * ci + n2 * ct
    dp = n2 * ci + n1 * ct
    if abs(ds) < 1e-15 or abs(dp) < 1e-15:
        return 1.0, 1.0, 0.0, 0.0
    rs = (n1 * ci - n2 * ct) / ds
    rp = (n2 * ci - n1 * ct) / dp
    Rs = np.clip(rs * rs, 0.0, 1.0)
    Rp = np.clip(rp * rp, 0.0, 1.0)
    return Rs, Rp, 1.0 - Rs, 1.0 - Rp

def refract(d, n, n1, n2):
    """Compute refracted ray direction."""
    Rs, Rp, Ts, Tp = fresnel_sp(d, n, n1, n2)
    η = n1 / n2
    ci = -np.dot(n, d)
    k = 1.0 - η * η * (1.0 - ci * ci)
    if k < 0.0:
        return None, True, Rs, Rp, Ts, Tp
    v_trans = norm(η*d + (η*ci - np.sqrt(k))*n)
    return v_trans, False, Rs, Rp, Ts, Tp

def reflect(v, n):
    """Compute reflected ray direction."""
    return norm(v - 2.0*np.dot(v, n)*n)

#Ray Tracing 
def trace_orientation_batch(verts, n_rays, max_depth, weight_cut):
    """Trace all rays for a given orientation using pool-based tracing."""
    y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
    if y1 <= y0:
        return []
    
    dy = (y1 - y0) / n_rays
    ys = y0 + (np.arange(n_rays) + 0.5) * dy
    x = np.min(verts[:, 0]) - 1e-6
    d0 = np.array([1.0, 0.0])
    #Weighting for each ray (L(⍺)/N); n_rays = 1000
    I0 = 1.0/n_rays  
    
    pool = [{"p": np.array([x, y]), "d": d0.copy(), "ws": I0, "wp": I0, "med": 0, "k": 0} for y in ys]
    out = []
    
    i = 0
    while i < len(pool):
        ray = pool[i]
        i += 1
        
        if (ray["ws"] + ray["wp"]) < weight_cut or ray["k"] > max_depth:
            continue
        
        n1 = 1.0 if ray["med"] == 0 else m_ice
        n2 = m_ice if ray["med"] == 0 else 1.0
        
        hit = first_hit(ray["p"], ray["d"], verts)
        if hit is None:
            if ray["med"] == 1:
                out.append((ray["d"], ray["ws"], ray["wp"]))
            continue
        
        hit_point, n_out = hit
        n_int = -n_out if ray["med"] == 0 else n_out
        if np.dot(n_int, ray["d"]) > 0.0:
            n_int = -n_int
        
        v_trans, tir, Rs, Rp, Ts, Tp = refract(ray["d"], n_int, n1, n2)
        
        # Reflection
        v_ref = reflect(ray["d"], n_int)
        ws_ref = ray["ws"] * Rs
        wp_ref = ray["wp"] * Rp
        if (ws_ref + wp_ref) > weight_cut:
            pool.append({"p": hit_point + 1e-9*v_ref, "d": v_ref, "ws": ws_ref, "wp": wp_ref, "med": ray["med"], "k": ray["k"] + 1})
        
        # Transmission
        if not tir:
            ws_tr = ray["ws"] * Ts
            wp_tr = ray["wp"] * Tp
            if (ws_tr + wp_tr) > weight_cut:
                if ray["med"] == 0:
                    pool.append({"p": hit_point + 1e-9*v_trans, "d": v_trans, "ws": ws_tr, "wp": wp_tr, "med": 1, "k": ray["k"] + 1})
                else:
                    out.append((v_trans, ws_tr, wp_tr))
    
    return out


def diffraction(theta, chi=500):
    """Compute diffraction contribution using numerical integration."""
    result = np.zeros_like(theta)
    
    alphas = np.linspace(0, np.pi/6, 500)
    da = alphas[1] - alphas[0]
    
    for i, th in enumerate(theta):
        integral = 0.0
        for a in alphas:
            arg = chi * np.cos(a) * np.sin(th)
            sinc = 1.0 if abs(arg) < 1e-8 else np.sin(arg) / arg
            integral += chi * np.cos(a) * sinc
        integral *= da
        result[i] = (1 / (2 * np.pi * chi)) * (1 + np.cos(th))**2 * integral**2
    
    return result


hs = np.zeros(n_bins)
hp = np.zeros(n_bins)

base = hexagon_vertices(a)
d0 = np.array([1.0, 0.0])

# Sample orientations
for i, th_deg in enumerate(np.linspace(0, 360.0, n_orientations, endpoint=False)):
    verts = rotate_pts(base, np.deg2rad(th_deg))
    rays_data = trace_orientation_batch(verts, n_rays, max_depth, weight_cut)
    
    for d, Is, Ip in rays_data:
        sc = np.clip(scatter_deg(d, d0), 0.0, 180.0)
        j = min(int(np.floor((sc / 180.0) * n_bins)), n_bins - 1)
        hs[j] += Is
        hp[j] += Ip
    
    if (i + 1) % 90 == 0:
        print(f"Processed {i + 1}/{n_orientations} orientations...")

hs /= n_orientations
hp /= n_orientations

h = hs + hp
angles = (np.arange(n_bins) + 0.5) * scatter_bin_deg
theta_rad = np.deg2rad(angles)

print(f"\nPhase function computed.")
print(f"Total intensity: {h.sum():.6f}")
print(f"hs sum: {hs.sum():.6f}, hp sum: {hp.sum():.6f}")

# Normalize: P_ag = (ΔF_j / Δθ_j) / Σ F_i
phase = (h / scatter_bin_deg) / h.sum()
phase_corrected = phase * n_orientations / h.sum()

# Diffraction contribution
phase_diff = diffraction(theta_rad)
# Normalize by integrating over the full radian range [0, π]
integral = trapezoid(phase_diff, theta_rad)
if integral > 0:
    phase_diff = phase_diff / integral
phase_combined = 0.5 * phase_corrected + 0.5 * phase_diff

# Calculate DoLP: -(P_parallel - P_perpendicular) / (P_perp + P_parallel)
dolp = np.zeros_like(h)
m = h > 0.0
dolp[m] = -(hp[m] - hs[m]) / h[m]

# ============================================================
# Plotting
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Phase function
ax = axes[0]
phase_log = np.copy(phase_combined)
phase_log[phase_log <= 0] = 1e-12
ax.plot(angles, phase_log, color='#1f4788', linewidth=2, label='Ray + Diffraction')
ax.set_yscale('log')
ax.set_xlabel(r'(Scattering angle) [°]')
ax.set_ylabel('$P_{11}$')
ax.set_title('Phase Function - Hexagonal Crystal')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(0, 180)
ax.legend()

# DoLP
ax = axes[1]
ax.plot(angles, dolp, 'g-', linewidth=2)
ax.set_xlabel('Scattering angle (degrees)')
ax.set_ylabel('$P_{12}$/P$_{11}$ (')
ax.set_title('Degree of Linear Polarization')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 180)
ax.set_ylim(-1.0, 1.0)

plt.tight_layout()
import os
save_path = os.path.expanduser("~/hexagon_phase_hybrid.png")
plt.savefig(save_path, dpi=150)
print(f"Figure saved to {save_path}")
plt.show()
