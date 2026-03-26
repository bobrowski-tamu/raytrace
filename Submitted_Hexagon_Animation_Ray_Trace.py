# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:45:58 2026

@author: jeff9
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# PARAMETERS
m = 1.31
max_bounces = 10
ray_position = 0.5
rotation = 0  #0.9

dt = 0.08
steps = 250


# HEXAGON
def hexagon_vertices(a=1.0, rotation=0.0):
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + rotation
    return np.stack((a*np.cos(angles), a*np.sin(angles)), axis=1)


# GEOMETRY
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


# FRESNEL
def fresnel(dir_in, n, n1, n2):
    cos_i = abs(np.dot(n, dir_in))
    sin_t2 = (n1/n2)**2 * (1 - cos_i**2)

    if sin_t2 > 1:
        return 1.0, 1.0, None

    cos_t = np.sqrt(1 - sin_t2)

    rs = ((n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t))**2
    rp = ((n1*cos_t - n2*cos_i)/(n1*cos_t + n2*cos_i))**2

    return rs, rp, cos_t


# OPTICS
def reflect(d, n):
    return d - 2*np.dot(d, n)*n

def refract(d, n, n1, n2):
    cos_i = -np.dot(n, d)
    eta = n1/n2
    k = 1 - eta**2*(1 - cos_i**2)
    if k < 0:
        return None
    return eta*d + (eta*cos_i - np.sqrt(k))*n


# RAY CLASS
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


# INITIAL STATE
verts = hexagon_vertices(1.0, rotation)
y = -1 + 2*ray_position

state = {
    "rays": [Ray(np.array([-2.0, y]), np.array([1.0, 0.0]), 1.0, m, 0, 'incident')],
    "segments": [],
    "exit_angles": [],
    "exit_weights": [],
    "far_segments": []
}


# STEP FUNCTION
def step_rays(state, verts):
    new_rays = []

    for ray in state["rays"]:
        if not ray.alive:
            continue

        if ray.depth >= max_bounces:
            ray.alive = False
            continue

        new_pos = ray.pos + ray.dir * dt

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

            rs, rp, cos_t = fresnel(ray.dir, n, ray.n1, ray.n2)
            R = 0.5 * (rs + rp)
            T = 1 - R

            if R > 1e-6:
                refl_dir = reflect(ray.dir, n)
                new_rays.append(Ray(hit + 1e-4*refl_dir, refl_dir,
                                    ray.n1, ray.n2, ray.depth+1,
                                    'reflect', ray.intensity * R))

            if cos_t is not None:
                refr_dir = refract(ray.dir, n, ray.n1, ray.n2)
                if refr_dir is not None and T > 1e-6:
                    new_rays.append(Ray(hit + 1e-4*refr_dir, refr_dir,
                                        ray.n2, ray.n1, ray.depth+1,
                                        'refract', ray.intensity * T))

        else:
            state["segments"].append((ray.pos.copy(), new_pos.copy(), ray.kind))
            ray.pos = new_pos

            if np.linalg.norm(ray.pos) > 2.5:
                theta = np.degrees(np.arctan2(ray.dir[1], ray.dir[0])) % 360
                state["exit_angles"].append(theta)
                state["exit_weights"].append(ray.intensity)

                origin = np.array([0.0, 0.0])
                far_point = origin + 2.5 * ray.dir
                state["far_segments"].append((origin, far_point))

                ray.alive = False

            new_rays.append(ray)

    state["rays"] = new_rays


# FIGURE SETUP
fig = plt.figure(figsize=(12,8))

ax_geom = plt.subplot2grid((2,2), (0,0))
ax_far  = plt.subplot2grid((2,2), (0,1))
ax_hist = plt.subplot2grid((2,2), (1,0), colspan=2)

# --- Geometry plot ---
lc_i = LineCollection([], colors='yellow', linewidths=2)
lc_r = LineCollection([], colors='blue', linewidths=1)
lc_t = LineCollection([], colors='red', linewidths=1)

ax_geom.add_collection(lc_i)
ax_geom.add_collection(lc_r)
ax_geom.add_collection(lc_t)

for i in range(len(verts)):
    p1 = verts[i]
    p2 = verts[(i+1)%len(verts)]
    ax_geom.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')

ax_geom.set_xlim(-3,3)
ax_geom.set_ylim(-3,3)
ax_geom.set_aspect('equal')
ax_geom.set_title("Crystal Interaction")

ax_geom.set_xticks([])
ax_geom.set_yticks([])
ax_geom.set_frame_on(False)

# --- Far-field plot ---
ax_far.set_xlim(-3,3)
ax_far.set_ylim(-3,3)
ax_far.set_aspect('equal')
ax_far.set_title("Far-Field Scattering")

ax_far.set_xticks([])
ax_far.set_yticks([])
ax_far.set_frame_on(False)

circle = plt.Circle((0,0), 2.5, color='black', fill=False)
ax_far.add_patch(circle)

for ang_deg in range(0,360,2):
    ang = np.radians(ang_deg)
    ax_far.plot([0,2.5*np.cos(ang)],
                [0,2.5*np.sin(ang)],
                color='gray', alpha=0.02)

for ang_deg in range(0,360,10):
    ang = np.radians(ang_deg)
    ax_far.text(2.7*np.cos(ang), 2.7*np.sin(ang),
                f"{ang_deg}°", fontsize=6,
                ha='center', va='center')

lc_far = LineCollection([], colors='green', linewidths=1)
ax_far.add_collection(lc_far)


# UPDATE
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

    # Far-field
    lc_far.set_segments(state["far_segments"])

    # Histogram
    ax_hist.clear()
    ax_hist.set_xlim(0,360)
    ax_hist.set_xlabel("Scattering Angle [°]")
    ax_hist.set_ylabel("Normalized Intensity")

    if len(state["exit_angles"]) > 0:
        hist, bins = np.histogram(state["exit_angles"],
                                  bins=72,
                                  weights=state["exit_weights"],
                                  range=(0,360))
        hist = hist / (np.sum(hist)+1e-12)
        ax_hist.bar((bins[:-1]+bins[1:])/2, hist, width=5)

    return lc_i, lc_r, lc_t, lc_far


# RUN
anim = FuncAnimation(fig, update, frames=steps, interval=30)

anim.save("ray_phase_fresnel.gif", writer="pillow", fps=20)

print("Saved: ray_phase_fresnel.gif")