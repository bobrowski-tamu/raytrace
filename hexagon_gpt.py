import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2D geometric-optics ray tracing for a regular hexagonal ice crystal
# using only numpy + matplotlib
#
# - Regular hexagon with circumradius a
# - Parallel rays incident from the left (+x direction)
# - Unpolarized Fresnel reflection/transmission
# - Full ray splitting at each interface
# - Up to max_interactions surface hits per branch
# - Rotation averaging over 0..360 deg
# - Collects:
#     1) azimuthal phase function: 0..360 deg
#     2) scattering-angle phase function: 0..180 deg
# ============================================================


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        return v.copy()
    return v / n


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def reflect(d, n):
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(d, n, n1, n2):
    """
    Vector Snell law.
    n points from medium 1 into medium 2.
    d is the incident unit vector.
    Returns None if total internal reflection occurs.
    """
    eta = n1 / n2
    cos_i = -np.dot(n, d)
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return None
    t = eta * d + (eta * cos_i - np.sqrt(k)) * n
    return normalize(t)


def fresnel_unpolarized(d, n, n1, n2):
    """
    Unpolarized Fresnel reflectance/transmittance.
    n points from medium 1 into medium 2.
    Returns (R, T).
    """
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
    T = 1.0 - R
    return R, T


def regular_hexagon_vertices(a):
    """
    Regular hexagon centered at origin.
    a = circumradius (radius of the imaginary circle around the hexagon)
    """
    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((a * np.cos(angles), a * np.sin(angles)))


def rotate_points(points, theta_rad):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def polygon_edges(vertices):
    n = len(vertices)
    return [(vertices[i], vertices[(i + 1) % n]) for i in range(n)]


def outward_normal(a, b):
    """
    For a CCW polygon, the outward normal of edge a->b is (dy, -dx).
    """
    e = b - a
    n = np.array([e[1], -e[0]])
    return normalize(n)


def ray_segment_intersection(ray_origin, ray_dir, seg_a, seg_b, eps=1e-12):
    """
    Solve:
        ray_origin + t * ray_dir = seg_a + u * (seg_b - seg_a)
    Returns (t, u) or None.
    """
    s = seg_b - seg_a
    denom = cross2(ray_dir, s)
    if abs(denom) < eps:
        return None

    qmp = seg_a - ray_origin
    t = cross2(qmp, s) / denom
    u = cross2(qmp, ray_dir) / denom

    if t > eps and -eps <= u <= 1.0 + eps:
        return t, u
    return None


def next_boundary_hit(pos, direction, vertices, eps=1e-10):
    """
    Find first intersection of ray with polygon boundary.
    Returns a dict or None.
    """
    best_t = np.inf
    best = None

    edges = polygon_edges(vertices)
    for i, (a, b) in enumerate(edges):
        hit = ray_segment_intersection(pos, direction, a, b, eps=eps)
        if hit is None:
            continue
        t, _ = hit
        if t < best_t:
            best_t = t
            best = {
                "point": pos + t * direction,
                "edge_index": i,
                "outward_normal": outward_normal(a, b),
                "distance": t,
            }

    return best


def azimuth_deg(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360.0


def scattering_angle_deg(v, incident_dir=np.array([1.0, 0.0])):
    c = np.clip(np.dot(normalize(v), normalize(incident_dir)), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def bin_angle_deg(angle_deg, n_bins, angle_max):
    if angle_max == 360.0:
        x = angle_deg % 360.0
    else:
        x = np.clip(angle_deg, 0.0, angle_max)

    idx = int(np.floor((x / angle_max) * n_bins))
    if idx == n_bins:
        idx = n_bins - 1
    return idx


def trace_one_orientation(
    vertices,
    n_rays=400,
    n_ice=1.31,
    n_air=1.0,
    max_interactions=15,
    intensity_cutoff=1e-6,
    eps_shift=1e-9,
):
    """
    Trace all rays for one particle orientation.

    A ray is stored as a dict with keys:
        pos, direction, intensity, inside, interactions

    Returns:
        exits = list of (direction, intensity)
    """
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])
    x_min = np.min(vertices[:, 0])

    band_height = y_max - y_min
    if band_height <= 0.0:
        return []

    dy = band_height / n_rays
    y_samples = y_min + (np.arange(n_rays) + 0.5) * dy

    x0 = x_min - 1e-6
    incoming_dir = np.array([1.0, 0.0])

    # Each incoming ray represents a strip of width dy in y
    initial_intensity = dy

    ray_pool = []
    exits = []

    for y0 in y_samples:
        ray_pool.append({
            "pos": np.array([x0, y0], dtype=float),
            "direction": incoming_dir.copy(),
            "intensity": initial_intensity,
            "inside": False,
            "interactions": 0,
        })

    i = 0
    while i < len(ray_pool):
        ray = ray_pool[i]
        i += 1

        if ray["intensity"] < intensity_cutoff:
            continue
        if ray["interactions"] >= max_interactions:
            continue

        hit = next_boundary_hit(ray["pos"], ray["direction"], vertices)
        if hit is None:
            continue

        p_hit = hit["point"]
        n_out = hit["outward_normal"]

        if ray["inside"]:
            # ice -> air
            n1, n2 = n_ice, n_air
            n_if = n_out.copy()
        else:
            # air -> ice
            n1, n2 = n_air, n_ice
            n_if = -n_out

        # Ensure interface normal opposes incoming direction
        if np.dot(n_if, ray["direction"]) > 0.0:
            n_if = -n_if

        R, T = fresnel_unpolarized(ray["direction"], n_if, n1, n2)

        # Reflected branch
        if R > 0.0:
            d_ref = reflect(ray["direction"], n_if)
            I_ref = ray["intensity"] * R
            pos_ref = p_hit + eps_shift * d_ref

            if ray["inside"]:
                ray_pool.append({
                    "pos": pos_ref,
                    "direction": d_ref,
                    "intensity": I_ref,
                    "inside": True,
                    "interactions": ray["interactions"] + 1,
                })
            else:
                exits.append((d_ref, I_ref))

        # Transmitted branch
        if T > 0.0:
            d_tr = refract(ray["direction"], n_if, n1, n2)
            if d_tr is not None:
                I_tr = ray["intensity"] * T
                pos_tr = p_hit + eps_shift * d_tr

                if ray["inside"]:
                    # exiting crystal
                    exits.append((d_tr, I_tr))
                else:
                    # entering crystal
                    ray_pool.append({
                        "pos": pos_tr,
                        "direction": d_tr,
                        "intensity": I_tr,
                        "inside": True,
                        "interactions": ray["interactions"] + 1,
                    })

    return exits


def simulate_phase_function(
    a=1.0,
    n_ice=1.31,
    n_rays=400,
    dtheta_deg=1.0,
    max_interactions=15,
    intensity_cutoff=1e-6,
    azimuth_bin_deg=1.0,
    scatter_bin_deg=1.0,
):
    base_hex = regular_hexagon_vertices(a)

    n_az_bins = int(round(360.0 / azimuth_bin_deg))
    n_sc_bins = int(round(180.0 / scatter_bin_deg))

    hist_az = np.zeros(n_az_bins, dtype=float)
    hist_sc = np.zeros(n_sc_bins, dtype=float)

    thetas_deg = np.arange(0.0, 360.0, dtheta_deg)

    for theta_deg in thetas_deg:
        verts = rotate_points(base_hex, np.deg2rad(theta_deg))

        exits = trace_one_orientation(
            verts,
            n_rays=n_rays,
            n_ice=n_ice,
            n_air=1.0,
            max_interactions=max_interactions,
            intensity_cutoff=intensity_cutoff,
        )

        for d_out, I_out in exits:
            az = azimuth_deg(d_out)
            sc = scattering_angle_deg(d_out)

            iaz = bin_angle_deg(az, n_az_bins, 360.0)
            isc = bin_angle_deg(sc, n_sc_bins, 180.0)

            hist_az[iaz] += I_out
            hist_sc[isc] += I_out

    if len(thetas_deg) > 0:
        hist_az /= len(thetas_deg)
        hist_sc /= len(thetas_deg)

    phase_az = hist_az / hist_az.sum() if hist_az.sum() > 0 else hist_az
    phase_sc = hist_sc / hist_sc.sum() if hist_sc.sum() > 0 else hist_sc

    az_centers = (np.arange(n_az_bins) + 0.5) * azimuth_bin_deg
    sc_centers = (np.arange(n_sc_bins) + 0.5) * scatter_bin_deg

    return {
        "azimuth_centers_deg": az_centers,
        "phase_azimuth": phase_az,
        "scatter_centers_deg": sc_centers,
        "phase_scatter": phase_sc,
        "raw_azimuth": hist_az,
        "raw_scatter": hist_sc,
    }


def plot_results(result):
    az_deg = result["azimuth_centers_deg"]
    p_az = result["phase_azimuth"]

    sc_deg = result["scatter_centers_deg"]
    p_sc = result["phase_scatter"]

    plt.figure(figsize=(10, 4))
    plt.plot(az_deg, p_az)
    plt.xlabel("Azimuth angle (deg)")
    plt.ylabel("Normalized intensity")
    plt.title("Azimuthal phase function")
    plt.xlim(0, 180)
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(sc_deg, p_sc)
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Normalized intensity")
    plt.title("Scattering-angle phase function")
    plt.xlim(0, 180)
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    result = simulate_phase_function(
        a=1.0,                # circumradius
        n_ice=1.31,
        n_rays=600,           # rays per orientation
        dtheta_deg=1.0,       # rotation step
        max_interactions=15,
        intensity_cutoff=1e-7,
        azimuth_bin_deg=1.0,
        scatter_bin_deg=1.0,
    )

    plot_results(result)

    # optional save
    np.savetxt(
        "phase_azimuth.txt",
        np.column_stack([result["azimuth_centers_deg"], result["phase_azimuth"]]),
        header="azimuth_deg phase_function"
    )

    np.savetxt(
        "phase_scattering_angle.txt",
        np.column_stack([result["scatter_centers_deg"], result["phase_scatter"]]),
        header="scattering_angle_deg phase_function"
    )