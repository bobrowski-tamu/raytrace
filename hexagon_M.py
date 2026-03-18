import numpy as np
import matplotlib.pyplot as plt

m = 1.31
N_ICE = m
HEX_RADIUS = 1.0
N_HITS = 4
FOLLOW_INTERNAL_REFLECTIONS = True


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        return v.copy()
    return v / n


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def hexagon_vertices(radius):
    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


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


def ray_polygon_hit(ray_origin, ray_dir, polygon_vertices):
    best_t = np.inf
    best = None

    for i in range(len(polygon_vertices)):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
        hit = ray_segment_intersection(ray_origin, ray_dir, p1, p2)
        if hit is None:
            continue
        t, _ = hit
        if t < best_t:
            best_t = t
            best = {
                "distance": t,
                "edge_index": i,
                "point": ray_origin + t * ray_dir,
                "normal_out": outward_normal(p1, p2),
            }

    return best


def orient_normal_toward_incoming(normal, incoming_dir):
    n = normal.copy()
    if np.dot(n, incoming_dir) > 0.0:
        n = -n
    return n


def plot_segment(p0, p1, color, label=None, lw=2):
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, label=label)


def main():
    vertices = hexagon_vertices(HEX_RADIUS)
    vertices_closed = np.vstack([vertices, vertices[0]])

    plt.figure(figsize=(7, 7))
    plt.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'b-', linewidth=2, label='Hexagon')

    # Incident ray from the left, traveling in +x (wavefront parallel to y-axis).
    ray_pos = np.array([-2.0, 0.5])
    ray_dir = normalize(np.array([1.0, 0.0]))
    inside_ice = False
    hit_points = []

    # Track one branch for readability: at each hit, show both split rays,
    # then continue tracing the transmitted branch.
    for hit_idx in range(1, N_HITS + 1):
        hit = ray_polygon_hit(ray_pos, ray_dir, vertices)
        if hit is None:
            break

        p_hit = hit['point']
        n_out = hit['normal_out']
        hit_points.append(p_hit.copy())

        if inside_ice:
            n1, n2 = N_ICE, 1.0
            n_if = n_out
        else:
            n1, n2 = 1.0, N_ICE
            n_if = -n_out

        n_if = orient_normal_toward_incoming(n_if, ray_dir)
        R, T = fresnel_unpolarized(ray_dir, n_if, n1, n2)
        d_ref = reflect(ray_dir, n_if)
        d_tr = refract(ray_dir, n_if, n1, n2)

        # Draw incoming segment up to this hit.
        color_in = 'orange' if hit_idx == 1 else 'green'
        label_in = 'Incident' if hit_idx == 1 else f'Path to hit {hit_idx}'
        plot_segment(ray_pos, p_hit, color=color_in, label=label_in)

        # Draw local reflected branch.
        plot_segment(
            p_hit,
            p_hit + 0.8 * d_ref,
            color='red',
            label=f'Reflected at hit {hit_idx}, R={R:.2f}'
        )

        # Draw interface normal at hit.
        plot_segment(
            p_hit,
            p_hit + 0.25 * n_if,
            color='black',
            label=f'Normal at hit {hit_idx}',
            lw=1.2,
        )

        # Draw a short transmitted direction marker.
        tr_color = 'cyan' if inside_ice else 'green'
        if d_tr is not None:
            plot_segment(
                p_hit,
                p_hit + 0.8 * d_tr,
                color=tr_color,
                label=f'Transmitted at hit {hit_idx}, T={T:.2f}'
            )

        # Choose which branch to continue tracing.
        if d_tr is None:
            # Total internal reflection: only reflected branch exists.
            ray_pos = p_hit + 1e-9 * d_ref
            ray_dir = d_ref
            continue

        if FOLLOW_INTERNAL_REFLECTIONS and inside_ice:
            # Keep the main path inside the crystal to generate many hits.
            ray_pos = p_hit + 1e-9 * d_ref
            ray_dir = d_ref
            inside_ice = True
        else:
            # Default: follow transmitted branch.
            ray_pos = p_hit + 1e-9 * d_tr
            ray_dir = d_tr
            inside_ice = not inside_ice

    # Plot all hit points found (up to N_HITS), with colors tied to N_HITS.
    if hit_points:
        cmap_colors = plt.cm.viridis(np.linspace(0.15, 0.95, N_HITS))
        for i, p in enumerate(hit_points):
            color_i = cmap_colors[min(i, N_HITS - 1)]
            plt.scatter(
                p[0],
                p[1],
                s=60,
                color=color_i,
                edgecolor='white',
                linewidth=0.8,
                zorder=6,
                label='Hit points' if i == 0 else None,
            )
            plt.text(
                p[0] + 0.03,
                p[1] + 0.03,
                f'{i + 1}',
                fontsize=8,
                color='black',
                zorder=7,
            )

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='0.5', linewidth=1)
    plt.axvline(0, color='0.5', linewidth=1)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=8)
    plt.title(f'Ray Splitting and Hits (N_HITS={N_HITS}, found={len(hit_points)})')
    plt.show()


if __name__ == '__main__':
    main()