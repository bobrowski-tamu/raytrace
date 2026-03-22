# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:00:59 2026

@author: jeff9
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# =============================
# PARAMETERS
# =============================
m = 1.31
max_bounces = 10
ray_position = 0.5
rotation = 0.25

dt = 0.08
steps = 250

# =============================
# HEXAGON
# =============================
def hexagon_vertices(a=1.0, rotation=0.0):
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + rotation
    return np.stack((a*np.cos(angles), a*np.sin(angles)), axis=1)

# =============================
# GEOMETRY
# =============================
def intersect(ray_origin, ray_dir, p1, p2):
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

def normal(p1, p2):
    e = p2 - p1
    n = np.array([-e[1], e[0]])
    return n / np.linalg.norm(n)

# =============================
# FRESNEL (NEW)
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
# OPTICS
# =============================
def reflect(d, n):
    return d - 2*np.dot(d, n)*n

def refract(d, n, n1, n2):
    cos_i = -np.dot(n, d)
    eta = n1/n2
    k = 1 - eta**2*(1 - cos_i**2)
    if k < 0:
        return None
    return eta*d + (eta*cos_i - np.sqrt(k))*n

# =============================
# RAY CLASS
# =============================
class Ray:
    def __init__(self, pos, direction, n1, n2, depth, kind, intensity=1.0):
        self.pos = pos
        self.dir = direction / np.linalg.norm(direction)
        self.n1 = n1
        self.n2 = n2
        self.depth = depth
        self.kind = kind
        self.intensity = intensity
        self.alive = True

# =============================
# INITIAL STATE
# =============================
verts = hexagon_vertices(1.0, rotation)
y = -1 + 2*ray_position

state = {
    "rays": [Ray(np.array([-2.0, y]), np.array([1.0, 0.0]), 1.0, m, 0, 'incident')],
    "segments": [],
    "exit_angles": [],
    "exit_weights": []
}

# =============================
# STEP FUNCTION
# =============================
def step_rays(state, verts):
    new_rays = []

    for ray in state["rays"]:
        if not ray.alive:
            continue

        # Kill at max bounce
        if ray.depth >= max_bounces:
            ray.alive = False
            continue

        new_pos = ray.pos + ray.dir * dt

        # Check intersections
        hit_data = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i+1)%len(verts)]
            hit = intersect(ray.pos, ray.dir, p1, p2)
            if hit is not None:
                dist = np.linalg.norm(hit - ray.pos)
                if dist < dt*1.5:
                    hit_data.append((hit, p1, p2))

        if hit_data:
            hit, p1, p2 = hit_data[0]
            n = normal(p1, p2)

            if np.dot(ray.dir, n) > 0:
                n = -n

            state["segments"].append((ray.pos.copy(), hit.copy(), ray.kind))

            # ===== FRESNEL SPLITTING =====
            rs, rp, cos_t = fresnel(ray.dir, n, ray.n1, ray.n2)
            R = 0.5 * (rs + rp)
            T = 1 - R

            # Reflection
            if R > 1e-6:
                refl_dir = reflect(ray.dir, n)
                new_rays.append(Ray(hit + 1e-4*refl_dir, refl_dir,
                                    ray.n1, ray.n2, ray.depth+1,
                                    'reflect', ray.intensity * R))

            # Refraction
            if cos_t is not None:
                refr_dir = refract(ray.dir, n, ray.n1, ray.n2)
                if refr_dir is not None and T > 1e-6:
                    new_rays.append(Ray(hit + 1e-4*refr_dir, refr_dir,
                                        ray.n2, ray.n1, ray.depth+1,
                                        'refract', ray.intensity * T))

        else:
            state["segments"].append((ray.pos.copy(), new_pos.copy(), ray.kind))
            ray.pos = new_pos

            # EXIT → record
            if np.linalg.norm(ray.pos) > 2.5:
                theta = np.degrees(np.arctan2(ray.dir[1], ray.dir[0])) % 360
                state["exit_angles"].append(theta)
                state["exit_weights"].append(ray.intensity)
                ray.alive = False

            new_rays.append(ray)

    state["rays"] = new_rays

# =============================
# FIGURE SETUP
# =============================
fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(12,6))

# Hexagon
for i in range(len(verts)):
    p1 = verts[i]
    p2 = verts[(i+1)%len(verts)]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')

# Radial grid
for ang_deg in range(0, 360, 2):
    ang = np.radians(ang_deg)
    ax.plot([0, 3*np.cos(ang)],
            [0, 3*np.sin(ang)],
            color='gray', alpha=0.02)

# Angle labels
for ang_deg in range(0, 360, 10):
    ang = np.radians(ang_deg)
    ax.text(3.2*np.cos(ang), 3.2*np.sin(ang),
            f"{ang_deg}°", fontsize=6,
            ha='center', va='center')

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal')
ax.set_title("Ray Splitting (Fresnel)")

# Line collections
lc_i = LineCollection([], colors='yellow', linewidths=2)
lc_r = LineCollection([], colors='blue', linewidths=1)
lc_t = LineCollection([], colors='red', linewidths=1)

ax.add_collection(lc_i)
ax.add_collection(lc_r)
ax.add_collection(lc_t)

# Histogram axis
ax_hist.set_xlim(0, 360)
ax_hist.set_xlabel("Scattering Angle (deg)")
ax_hist.set_ylabel("Normalized Intensity")
ax_hist.set_title("Phase Function Build")

# =============================
# UPDATE
# =============================
def update(frame):
    step_rays(state, verts)

    lines_i, lines_r, lines_t = [], [], []

    for p1, p2, kind in state["segments"]:
        if kind == 'incident':
            lines_i.append([p1, p2])
        elif kind == 'reflect':
            lines_r.append([p1, p2])
        else:
            lines_t.append([p1, p2])

    lc_i.set_segments(lines_i)
    lc_r.set_segments(lines_r)
    lc_t.set_segments(lines_t)

    # Histogram
    ax_hist.clear()
    ax_hist.set_xlim(0, 360)
    ax_hist.set_xlabel("Scattering Angle (deg)")
    ax_hist.set_ylabel("Normalized Intensity")

    if len(state["exit_angles"]) > 0:
        hist, bins = np.histogram(state["exit_angles"],
                                  bins=72,
                                  weights=state["exit_weights"],
                                  range=(0,360))

        hist = hist / (np.sum(hist) + 1e-12)

        ax_hist.bar((bins[:-1]+bins[1:])/2, hist, width=5)

    return lc_i, lc_r, lc_t

# =============================
# RUN
# =============================
anim = FuncAnimation(fig, update, frames=steps, interval=30)

anim.save("ray_phase_fresnel.gif", writer="pillow", fps=20)

print("Saved: ray_phase_fresnel.gif")