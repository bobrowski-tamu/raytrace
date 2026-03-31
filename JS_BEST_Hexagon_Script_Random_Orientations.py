import numpy as np
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

# PARAMETERS
m = 1.31                # Refractive index of ice
num_rays = 1000         # Number of parallel rays
num_angles = 720        # Angular resolution (bins)
num_rot = 100           # Number of random rotation steps
max_bounces = 10        # Maximum internal reflections

# HEXAGON GEOMETRY WITH OUTWARD NORMALS
def create_hexagon(a=1.0, alpha=0.0):
    """
    Create hexagon vertices and outward-pointing unit normals.
    alpha: rotation angle in radians
    Returns: vertices (6x2), normals (6x2)
    """
    # Rotation matrix
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])
    
    # Base vertices (flat-top hexagon)
    base_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    vertices = np.zeros((6, 2))
    for i, ang in enumerate(base_angles):
        vertices[i] = a * np.array([np.cos(ang), np.sin(ang)])
    
    # Rotate vertices
    vertices = (R @ vertices.T).T
    
    # Compute outward normals for each face
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        # Perpendicular to edge
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        # Ensure outward pointing (away from center)
        midpoint = (p1 + p2) / 2
        if np.dot(n, midpoint) < 0:
            n = -n
        normals[i] = n
    
    return vertices, normals

# RAY-LINE INTERSECTION
def ray_line_intersection(ray_origin, ray_dir, p1, p2, normal):
    """
    Find intersection of ray with line segment using parametric equations.
    """
    denom = np.dot(ray_dir, normal)
    if abs(denom) < 1e-12:
        return None, None  # Ray parallel to face
    
    t = np.dot(p1 - ray_origin, normal) / denom
    if t < 1e-9:
        return None, None  # Behind ray origin
    
    hit_point = ray_origin + t * ray_dir
    
    # Check if hit point is within segment
    edge = p2 - p1
    edge_len = np.linalg.norm(edge)
    edge_unit = edge / edge_len
    proj = np.dot(hit_point - p1, edge_unit)
    
    if 0 <= proj <= edge_len:
        return hit_point, t
    return None, None

# FRESNEL COEFFICIENTS (2D: TE and TM polarizations)
def fresnel_coefficients(cos_i, n1, n2):
    cos_i = abs(cos_i)
    sin_i2 = 1 - cos_i**2
    sin_t2 = (n1/n2)**2 * sin_i2
    
    if sin_t2 > 1.0:
        # Total internal reflection
        return 1.0, 1.0, None
    
    cos_t = np.sqrt(1 - sin_t2)
    
    # Fresnel equations
    Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))**2
    Rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i))**2
    
    return Rs, Rp, cos_t

# SNELL'S LAW REFRACTION
def snell_refract(dir_in, normal, n1, n2):
    """
    Apply Snell's law to get refracted direction.
    dir_in: incident ray direction (unit vector)
    normal: surface normal (pointing into medium ray is leaving)
    """
    cos_i = -np.dot(dir_in, normal)
    if cos_i < 0:
        normal = -normal
        cos_i = -cos_i
    
    eta = n1 / n2
    sin_t2 = eta**2 * (1 - cos_i**2)
    
    if sin_t2 > 1.0:
        return None  # TIR
    
    cos_t = np.sqrt(1 - sin_t2)
    refr_dir = eta * dir_in + (eta * cos_i - cos_t) * normal
    return refr_dir / np.linalg.norm(refr_dir)

# REFLECTION
def reflect(dir_in, normal):
    """Reflect direction about normal."""
    return dir_in - 2 * np.dot(dir_in, normal) * normal

# TRACE RAY THROUGH CRYSTAL
def trace_ray(origin, direction, vertices, normals, n1, n2, depth=0, I_s=0.5, I_p=0.5):
    """
    Recursively trace ray through crystal.
    Returns list of (exit_direction, I_s, I_p) tuples.
    """
    if depth > max_bounces or (I_s + I_p) < 1e-10:
        return []
    
    direction = direction / np.linalg.norm(direction)
    
    # Find closest intersection
    min_t = np.inf
    hit_point = None
    hit_face = None
    
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        point, t = ray_line_intersection(origin, direction, p1, p2, normals[i])
        if point is not None and t < min_t:
            min_t = t
            hit_point = point
            hit_face = i
    
    if hit_point is None:
        # Ray exits crystal
        return [(direction, I_s, I_p)]
    
    # Get normal at hit point
    n = normals[hit_face]
    
    # Ensure normal points against ray direction
    if np.dot(direction, n) > 0:
        n = -n
    
    cos_i = abs(np.dot(direction, n))
    Rs, Rp, cos_t = fresnel_coefficients(cos_i, n1, n2)
    
    rays_out = []
    
    # Reflected ray (stays in same medium)
    refl_dir = reflect(direction, n)
    refl_origin = hit_point + 1e-9 * refl_dir
    rays_out += trace_ray(refl_origin, refl_dir, vertices, normals, 
                          n1, n2, depth + 1, I_s * Rs, I_p * Rp)
    
    # Refracted ray (if not TIR)
    if cos_t is not None:
        refr_dir = snell_refract(direction, n, n1, n2)
        if refr_dir is not None:
            refr_origin = hit_point + 1e-9 * refr_dir
            # Swap n1 and n2 for the refracted ray
            rays_out += trace_ray(refr_origin, refr_dir, vertices, normals,
                                  n2, n1, depth + 1, I_s * (1 - Rs), I_p * (1 - Rp))
    
    return rays_out

# FIRST ENTRY INTO CRYSTAL
def enter_crystal(origin, direction, vertices, normals, n_out=1.0, n_in=1.31):
    """
    Handle first entry of ray into the crystal.
    """
    direction = direction / np.linalg.norm(direction)
    
    min_t = np.inf
    hit_point = None
    hit_face = None
    
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        point, t = ray_line_intersection(origin, direction, p1, p2, normals[i])
        if point is not None and t < min_t:
            min_t = t
            hit_point = point
            hit_face = i
    
    if hit_point is None:
        return []  # Ray misses crystal
    
    n = normals[hit_face]
    if np.dot(direction, n) > 0:
        n = -n
    
    cos_i = abs(np.dot(direction, n))
    Rs, Rp, cos_t = fresnel_coefficients(cos_i, n_out, n_in)
    
    rays_out = []
    
    # Refracted ray enters crystal
    if cos_t is not None:
        refr_dir = snell_refract(direction, n, n_out, n_in)
        if refr_dir is not None:
            refr_origin = hit_point + 1e-9 * refr_dir
            I_s = 0.5 * (1 - Rs)
            I_p = 0.5 * (1 - Rp)
            rays_out += trace_ray(refr_origin, refr_dir, vertices, normals,
                                  n_in, n_out, depth=1, I_s=I_s, I_p=I_p)
    
    return rays_out

# MAIN SIMULATION WITH RANDOM ORIENTATIONS
def simulate():
    """
    Run the ray tracing simulation with RANDOM hexagon orientations.
    Uses random angles 0 to 2π for fully random crystal orientations.
    """
    # Angular bins for phase function (0 to 180 degrees)
    theta_bins = np.linspace(0, np.pi, num_angles + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    
    I_s_total = np.zeros(num_angles)
    I_p_total = np.zeros(num_angles)
    
    # RANDOM rotation angles (0 to 2π)
    rng = np.random.default_rng(seed=42)  # seed for reproducibility
    alphas = rng.uniform(0, 2*np.pi, num_rot)
    
    # Impact parameters (where rays hit)
    y_positions = np.linspace(-1.5, 1.5, num_rays)
    dy = y_positions[1] - y_positions[0] if num_rays > 1 else 3.0
    
    for idx, alpha in enumerate(alphas):
        vertices, normals = create_hexagon(a=1.0, alpha=alpha)
        
        for y in y_positions:
            origin = np.array([-3.0, y])
            direction = np.array([1.0, 0.0])
            
            # Trace ray through crystal
            exit_rays = enter_crystal(origin, direction, vertices, normals, 1.0, m)
            
            for d, Is, Ip in exit_rays:
                # Calculate scattering angle
                d_norm = d / np.linalg.norm(d)
                cos_angle = np.clip(d_norm[0], -1, 1)
                
                angle = np.arccos(cos_angle)
                
                # Find bin index
                idx_bin = np.searchsorted(theta_bins, angle) - 1
                idx_bin = np.clip(idx_bin, 0, num_angles - 1)
                
                # Weight by impact parameter
                weight = 1.0
                I_s_total[idx_bin] += weight * Is
                I_p_total[idx_bin] += weight * Ip
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{num_rot} random orientations...")
    
    # Normalize
    P_total = I_s_total + I_p_total
    
    # Proper 2D normalization
    d_theta = theta_centers[1] - theta_centers[0]
    norm_factor = np.sum(P_total) * d_theta / np.pi
    if norm_factor > 0:
        P_total = P_total / norm_factor
        I_s_total = I_s_total / norm_factor
        I_p_total = I_p_total / norm_factor
    
    return theta_centers, P_total, I_s_total, I_p_total

# DIFFRACTION CORRECTION (ANALYTICAL APPROXIMATION)
def diffraction_pattern(theta, chi=500):
    """
    Diffraction approximation for 2D hexagonal crystal.
    Uses analytical approximation valid for large chi.
    """
    result = np.zeros_like(theta)
    
    for i, th in enumerate(theta):
        sin_th = np.sin(th)
        
        if sin_th < 1e-10:
            integral = chi / 2
        else:
            x_max = chi * sin_th
            if x_max > 1:
                integral = np.sqrt(np.pi / (2 * chi * sin_th)) * chi
            else:
                integral = chi * (np.pi/6) * (1 - (chi * sin_th)**2 / 6)
        
        result[i] = (1.0 / (2 * np.pi * chi)) * (1 + np.cos(th))**2 * integral
    
    # Normalize
    d_theta = theta[1] - theta[0]
    norm = np.sum(result) * d_theta / np.pi
    if norm > 0:
        result /= norm
    
    return result

# RUN SIMULATION
print("Starting simulation with RANDOM orientations...")
theta, P_ray, I_s, I_p = simulate()
print("Ray tracing complete.")

# Calculate degree of linear polarization
DoLP = (I_s - I_p) / (I_s + I_p + 1e-12)

# Add diffraction contribution
P_diff = diffraction_pattern(theta, chi=500)

# Combine ray tracing and diffraction
P_total = 0.5 * P_ray + 0.5 * P_diff

# Smooth the phase function to reduce noise
def smooth(y, window=5):
    """Simple moving average smoothing."""
    if window < 2:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1
    return np.concatenate([y[:pad_left], smoothed, y[-(pad_right):]])

P_smooth = smooth(P_total, window=7)

# PLOTTING
theta_deg = np.degrees(theta)

# Phase Function Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_deg, P_total, 'b-', alpha=1, label='Ray + Diffraction (Random Orientations)')
ax.set_yscale('log')
ax.set_ylim(1e-3, 1e3)
ax.set_xlim(0, 180)
ax.set_xlabel('Scattering Angle [°]', fontsize=12)
ax.set_ylabel('Phase Function (P11)', fontsize=12)
ax.set_title('2D Hexagonal Ice Crystal Phase Function (RANDOM Orientations)', fontsize=14)
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# Polarization Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_deg, DoLP, 'b-', linewidth=1.5)
ax.set_xlim(0, 180)
ax.set_ylim(-1, 1)
ax.set_xlabel('Scattering Angle [°]', fontsize=12)
ax.set_ylabel('Degree of Linear Polarization (-P12/P11)', fontsize=12)
ax.set_title('Polarization (Random Orientations)', fontsize=14)
ax.grid(True, which='both', alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=1)
plt.tight_layout()
plt.show()

elapsed = time.perf_counter() - start
print(f"\nElapsed time: {elapsed:.2f} seconds")
