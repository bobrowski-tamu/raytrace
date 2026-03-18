import numpy as np
import matplotlib.pyplot as plt

m = 1.31
n_rays = 1000
dtheta = 1.0
n_interact = 10
a = 1

def hexagon_vertices(a):
    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((a * np.cos(angles), a * np.sin(angles)))

def hexagon_edges(a):
    v = hexagon_vertices(a)
    edges = []

    for i in range(len(v)):
        p1 = v[i]
        p2 = v[(i + 1) % len(v)]   # wraps last vertex back to first
        edges.append((p1, p2))

    return edges


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        return v.copy()
    return v / n


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def outward_normal(p1, p2):
    e = p2 - p1
    return normalize(np.array([e[1], -e[0]]))


def reflect(d, n):
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(d, n, n1, n2):
    eta = n1 / n2
    cos_i = -np.dot(n, d)
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return None
    t = eta * d + (eta * cos_i - np.sqrt(k)) * n
    return normalize(t)


def fresnel_unpolarized(d, n, n1, n2):
    cos_i = -np.dot(n, d)
    cos_i = np.clip(cos_i, -1.0, 1.0)

    eta = n1 / n2
    sin_t2 = eta**2 * max(0.0, 1.0 - cos_i**2)
    if sin_t2 > 1.0:
        return 1.0, 0.0

    cos_t = np.sqrt(max(0.0, 1.0 - sin_t2))

    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n2 * cos_i + n1 * cos_t
    if abs(denom_s) < 1e-15 or abs(denom_p) < 1e-15:
        return 1.0, 0.0

    rs = (n1 * cos_i - n2 * cos_t) / denom_s
    rp = (n2 * cos_i - n1 * cos_t) / denom_p
    R = 0.5 * (rs**2 + rp**2)
    R = np.clip(R, 0.0, 1.0)
    return R, 1.0 - R


def ray_segment_intersection(ray_origin, ray_dir, p1, p2, eps=1e-12):
    s = p2 - p1
    denom = cross2(ray_dir, s)
    if abs(denom) < eps:
        return None

    qmp = p1 - ray_origin
    t = cross2(qmp, s) / denom
    u = cross2(qmp, ray_dir) / denom

    if t > eps and -eps <= u <= 1.0 + eps:
        return t, u
    return None


def first_hit(ray_origin, ray_dir, vertices):
    best_t = np.inf
    best = None

    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        hit = ray_segment_intersection(ray_origin, ray_dir, p1, p2)
        if hit is None:
            continue
        t, _ = hit
        if t < best_t:
            best_t = t
            best = {
                "point": ray_origin + t * ray_dir,
                "normal_out": outward_normal(p1, p2),
            }

    return best

vertices = hexagon_vertices(a)

plt.figure()

# Close the hexagon by appending the first vertex at the end
vertices_closed = np.vstack([vertices, vertices[0]])

# Plot the hexagon outline
plt.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'b-', linewidth=2)

# Incident ray: from the left, traveling to the right (+x).
# (Its wavefront is parallel to y-axis.)
ray_origin = np.array([-2.0, 0.5])
ray_dir = normalize(np.array([1.0, 0.0]))

hit = first_hit(ray_origin, ray_dir, vertices)
if hit is None:
    raise RuntimeError("Ray did not hit the hexagon.")

p_hit = hit["point"]
n_out = hit["normal_out"]

# For air -> ice, interface normal points into the crystal.
n_if = -n_out
if np.dot(n_if, ray_dir) > 0.0:
    n_if = -n_if

ray_ref = reflect(ray_dir, n_if)
ray_tr = refract(ray_dir, n_if, 1.0, m)
R1, T1 = fresnel_unpolarized(ray_dir, n_if, 1.0, m)

# Draw incident ray segment up to intersection point
plt.plot([ray_origin[0], p_hit[0]], [ray_origin[1], p_hit[1]], color='orange', linewidth=2, label='Incident')

# Draw reflected ray
L_ref = 1.6
ref_end = p_hit + L_ref * ray_ref
plt.plot([p_hit[0], ref_end[0]], [p_hit[1], ref_end[1]], color='red', linewidth=2, label=f'Reflected R={R1:.2f}')

# Draw refracted ray (Snell)
if ray_tr is not None:
    # Trace inside-ice ray to next boundary.
    eps_shift = 1e-9
    hit2 = first_hit(p_hit + eps_shift * ray_tr, ray_tr, vertices)
    if hit2 is not None:
        p_hit2 = hit2["point"]

        # Inside segment from first to second boundary.
        plt.plot([p_hit[0], p_hit2[0]], [p_hit[1], p_hit2[1]], color='green', linewidth=2, label=f'Transmitted in ice T={T1:.2f}')

        n_out2 = hit2["normal_out"]
        n_if2 = n_out2.copy()  # ice -> air normal
        if np.dot(n_if2, ray_tr) > 0.0:
            n_if2 = -n_if2

        ray_ref2 = reflect(ray_tr, n_if2)
        ray_tr2 = refract(ray_tr, n_if2, m, 1.0)
        R2, T2 = fresnel_unpolarized(ray_tr, n_if2, m, 1.0)

        # Reflected branch at second boundary (stays inside ice)
        L_ref2 = 1.1
        ref2_end = p_hit2 + L_ref2 * ray_ref2
        plt.plot(
            [p_hit2[0], ref2_end[0]],
            [p_hit2[1], ref2_end[1]],
            color='magenta',
            linewidth=2,
            label=f'Second reflected (ice) R={R2:.2f}'
        )

        # Transmitted branch at second boundary (exits to air)
        if ray_tr2 is not None:
            L_tr2 = 1.6
            tr2_end = p_hit2 + L_tr2 * ray_tr2
            plt.plot(
                [p_hit2[0], tr2_end[0]],
                [p_hit2[1], tr2_end[1]],
                color='cyan',
                linewidth=2,
                label=f'Second transmitted (air) T={T2:.2f}'
            )
        else:
            plt.scatter([p_hit2[0]], [p_hit2[1]], color='cyan', s=40, label='Total internal reflection')
    else:
        L_tr = 1.1
        tr_end = p_hit + L_tr * ray_tr
        plt.plot([p_hit[0], tr_end[0]], [p_hit[1], tr_end[1]], color='green', linewidth=2, label=f'Transmitted in ice T={T1:.2f}')

# Draw surface normal at hit point
L_n = 0.35
normal_end = p_hit + L_n * n_if
plt.plot([p_hit[0], normal_end[0]], [p_hit[1], normal_end[1]], 'k--', linewidth=1.5, label='Interface normal')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0)
plt.axvline(0)
plt.grid()
plt.gca().set_aspect('equal')
plt.legend()

plt.show()