# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:04:36 2026

@author: jeff9
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

# =============================
# PARAMETERS
# =============================
m = 1.31
num_rays = 100
num_angles = 360
num_rot = 100
max_bounces = 15   #  USER CONTROL: number of internal reflections

# =============================
# HEXAGON GEOMETRY
# =============================
def hexagon_vertices(a=1.0, rotation=0.0):
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + rotation
    return np.stack((a*np.cos(angles), a*np.sin(angles)), axis=1)

# =============================
# INTERSECTION
# =============================
def intersect_ray_segment(ray_origin, ray_dir, p1, p2):
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-10:
        return None

    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    if t1 > 1e-6 and 0 <= t2 <= 1:
        return ray_origin + t1 * ray_dir
    return None

# =============================
# NORMAL
# =============================
def normal(p1, p2):
    edge = p2 - p1
    n = np.array([-edge[1], edge[0]])
    return n / np.linalg.norm(n)

# =============================
# FRESNEL COEFFICIENTS
# =============================
def fresnel(dir_in, n, n1, n2):
    cos_i = abs(np.dot(n, dir_in))
    sin_t2 = (n1/n2)**2 * (1 - cos_i**2)

    if sin_t2 > 1:
        return 1.0, 1.0, None  # total internal reflection

    cos_t = np.sqrt(1 - sin_t2)

    rs = ((n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t))**2
    rp = ((n1*cos_t - n2*cos_i)/(n1*cos_t + n2*cos_i))**2

    return rs, rp, cos_t

# =============================
# REFLECT / REFRACT
# =============================
def reflect(dir_in, n):
    return dir_in - 2*np.dot(dir_in, n)*n


def refract(dir_in, n, n1, n2):
    cos_i = -np.dot(n, dir_in)
    eta = n1/n2
    k = 1 - eta**2*(1 - cos_i**2)
    if k < 0:
        return None
    return eta*dir_in + (eta*cos_i - np.sqrt(k))*n

# =============================
# TRACE WITH RECURSION
# =============================
def trace_ray(origin, direction, vertices, n1, n2, depth=0, I_perp=0.5, I_par=0.5):
    if depth > max_bounces:
        return []

    hits = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i+1)%len(vertices)]
        hit = intersect_ray_segment(origin, direction, p1, p2)
        if hit is not None:
            hits.append((hit, i))

    if not hits:
        return [(direction, I_perp, I_par)]

    hit, idx = min(hits, key=lambda x: np.linalg.norm(x[0]-origin))
    p1 = vertices[idx]
    p2 = vertices[(idx+1)%len(vertices)]
    n = normal(p1, p2)

    if np.dot(direction, n) > 0:
        n = -n
    
    rs, rp, cos_t = fresnel(direction, n, n1, n2)

    rays_out = []

    # Reflection
    refl_dir = reflect(direction, n)
    rays_out += trace_ray(hit + 1e-6*refl_dir, refl_dir, vertices, n1, n2,
                          depth+1, I_perp*rs, I_par*rp)

    # Refraction
    if cos_t is not None:
        refr_dir = refract(direction, n, n1, n2)
        if refr_dir is not None:
            rays_out += trace_ray(hit + 1e-6*refr_dir, refr_dir, vertices, n2, n1,
                                  depth+1, I_perp*(1-rs), I_par*(1-rp))

    return rays_out

# =============================
# DIFFRACTION (YOUR FORMULA)
# =============================
def diffraction(theta):
    chi = 500
    result = np.zeros_like(theta)

    alphas = np.linspace(0, np.pi/6, 200)

    for i, th in enumerate(theta):
        integral = 0
        for a in alphas:
            x = chi*np.cos(a)*np.sin(th)
            if abs(x) < 1e-8:
                sinc = 1
            else:
                sinc = np.sin(x)/x
            integral += chi*np.cos(a)*sinc
        integral *= (alphas[1]-alphas[0])
        result[i] = (1/(2*np.pi*chi))*(1+np.cos(th))**2 * integral

    return result

# =============================
# SIMULATION
# =============================
def simulate():
    theta_bins = np.linspace(0, np.pi, num_angles)
    I_perp = np.zeros_like(theta_bins)
    I_par = np.zeros_like(theta_bins)

    for r in np.linspace(0, np.pi/6, num_rot):
        verts = hexagon_vertices(1.0, rotation=r)

        ys = np.linspace(-1, 1, num_rays)
        for y in ys:
            origin = np.array([-2.0, y])
            direction = np.array([1.0, 0.0])

            rays = trace_ray(origin, direction, verts, 1.0, m)

            for d, Ip, Ia in rays:
                angle = np.arccos(d[0]/np.linalg.norm(d))
                idx = np.argmin(np.abs(theta_bins - angle))
                #
                weight = 1.0 / num_rays
                I_perp[idx] += weight * Ip
                I_par[idx] += weight * Ia
                #
                #I_perp[idx] += Ip
                #I_par[idx] += Ia

    return theta_bins, I_perp, I_par

# =============================
# RUN
# =============================
theta, I_perp, I_par = simulate()

# Phase function
P_ray = I_perp + I_par
#
P_ray = P_ray / (np.sin(theta) + 1e-6)
#
#P_ray /= np.trapz(P_ray, theta) / np.pi

# Polarization
DoLP = (I_perp - I_par) / (I_perp + I_par + 1e-12)

# Diffraction
P_diff = diffraction(theta)
P_diff /= np.trapz(P_diff, theta) / np.pi

# Combine
P_total = 0.5*P_ray + 0.5*P_diff

# =============================
# PLOT
# =============================
# Convert to degrees for plotting
theta_deg = np.degrees(theta)

# Phase function (log scale)
plt.figure()
plt.plot(theta_deg, P_total)
plt.yscale('log')
plt.ylim(1e-3, 1e3)
plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Phase Function')
plt.title('Phase Function (Hexagonal Crystal)')
plt.show()

# Polarization (keep linear scale, but in degrees)
plt.figure()
plt.plot(theta_deg, DoLP)
plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Degree of Linear Polarization')
plt.title('Polarization')
plt.show()

elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.6f} seconds")