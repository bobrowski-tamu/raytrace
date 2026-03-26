import numpy as np
import matplotlib.pyplot as plt


# 2D geometric-optics scattering by a regular hexagon
# Output: phase function and DoLP vs scattering angle (0..180 deg)


def norm(v):
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def reflect(d, n):
    return norm(d - 2.0 * np.dot(d, n) * n)


def refract(d, n, n1, n2):
    eta = n1 / n2
    ci = -np.dot(n, d)
    k = 1.0 - eta * eta * (1.0 - ci * ci)
    if k < 0.0:
        return None
    return norm(eta * d + (eta * ci - np.sqrt(k)) * n)


def fresnel_sp(d, n, n1, n2):
    ci = np.clip(-np.dot(n, d), -1.0, 1.0)
    eta = n1 / n2
    st2 = eta * eta * max(0.0, 1.0 - ci * ci)
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


def hex_vertices(a):
    ang = np.deg2rad(np.arange(0.0, 360.0, 60.0))
    return np.column_stack((a * np.cos(ang), a * np.sin(ang)))


def rotate_pts(pts, th):
    c, s = np.cos(th), np.sin(th)
    return pts @ np.array([[c, -s], [s, c]]).T


def first_hit(pos, d, verts, eps=1e-10):
    best_t, best = np.inf, None
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        s = b - a
        den = cross2(d, s)
        if abs(den) < eps:
            continue
        q = a - pos
        t = cross2(q, s) / den
        u = cross2(q, d) / den
        if t > eps and -eps <= u <= 1.0 + eps and t < best_t:
            e = b - a
            n_out = norm(np.array([e[1], -e[0]]))
            best_t = t
            best = (pos + t * d, n_out)
    return best


def scatter_deg(v, inc=np.array([1.0, 0.0])):
    c = np.clip(np.dot(norm(v), norm(inc)), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def trace_orientation(verts, n_rays=400, n_ice=1.31, n_air=1.0, max_hits=15, cutoff=1e-6, eps=1e-9):
    y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
    if y1 <= y0:
        return []
    dy = (y1 - y0) / n_rays
    ys = y0 + (np.arange(n_rays) + 0.5) * dy
    x = np.min(verts[:, 0]) - 1e-6
    d0 = np.array([1.0, 0.0])
    I0 = 0.5 * dy  # unpolarized: split equally into s/p

    pool = [{"p": np.array([x, y]), "d": d0.copy(), "Is": I0, "Ip": I0, "in": False, "k": 0} for y in ys]
    out = []

    i = 0
    while i < len(pool):
        r = pool[i]
        i += 1
        if r["Is"] + r["Ip"] < cutoff or r["k"] >= max_hits:
            continue

        hit = first_hit(r["p"], r["d"], verts)
        if hit is None:
            continue
        ph, n_out = hit

        if r["in"]:
            n1, n2, n_if = n_ice, n_air, n_out.copy()
        else:
            n1, n2, n_if = n_air, n_ice, -n_out
        if np.dot(n_if, r["d"]) > 0.0:
            n_if = -n_if

        Rs, Rp, Ts, Tp = fresnel_sp(r["d"], n_if, n1, n2)

        d_ref = reflect(r["d"], n_if)
        Is_ref, Ip_ref = r["Is"] * Rs, r["Ip"] * Rp
        if Is_ref + Ip_ref > 0.0:
            p_ref = ph + eps * d_ref
            if r["in"]:
                pool.append({"p": p_ref, "d": d_ref, "Is": Is_ref, "Ip": Ip_ref, "in": True, "k": r["k"] + 1})
            else:
                out.append((d_ref, Is_ref, Ip_ref))

        d_tr = refract(r["d"], n_if, n1, n2)
        Is_tr, Ip_tr = r["Is"] * Ts, r["Ip"] * Tp
        if d_tr is not None and Is_tr + Ip_tr > 0.0:
            p_tr = ph + eps * d_tr
            if r["in"]:
                out.append((d_tr, Is_tr, Ip_tr))
            else:
                pool.append({"p": p_tr, "d": d_tr, "Is": Is_tr, "Ip": Ip_tr, "in": True, "k": r["k"] + 1})

    return out


def simulate(a=1.0, n_ice=1.31, n_rays=400, dtheta_deg=1.0, max_hits=15, cutoff=1e-6, scatter_bin_deg=1.0):
    base = hex_vertices(a)
    n_bins = int(round(180.0 / scatter_bin_deg))
    hs = np.zeros(n_bins)
    hp = np.zeros(n_bins)
    thetas = np.arange(0.0, 360.0, dtheta_deg)

    for th in thetas:
        verts = rotate_pts(base, np.deg2rad(th))
        for d, Is, Ip in trace_orientation(verts, n_rays=n_rays, n_ice=n_ice, max_hits=max_hits, cutoff=cutoff):
            sc = np.clip(scatter_deg(d), 0.0, 180.0)
            j = min(int(np.floor((sc / 180.0) * n_bins)), n_bins - 1)
            hs[j] += Is
            hp[j] += Ip

    if len(thetas) > 0:
        hs /= len(thetas)
        hp /= len(thetas)

    h = hs + hp
    phase = h / h.sum() if h.sum() > 0.0 else h
    dolp = np.zeros_like(h)
    m = h > 0.0
    dolp[m] = (hs[m] - hp[m]) / h[m]
    ang = (np.arange(n_bins) + 0.5) * scatter_bin_deg

    return {
        "scatter_centers_deg": ang,
        "phase_scatter": phase,
        "dolp_scatter": dolp,
        "raw_scatter": h,
        "raw_scatter_s": hs,
        "raw_scatter_p": hp,
    }


def plot_results(r):
    sc, p, q = r["scatter_centers_deg"], r["phase_scatter"], r["dolp_scatter"]

    plt.figure(figsize=(10, 4))
    plt.plot(sc, p)
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Normalized intensity")
    plt.title("Scattering-angle phase function")
    plt.xlim(0, 180)
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(sc, q)
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("DoLP")
    plt.title("Degree of linear polarization")
    plt.xlim(0, 180)
    plt.ylim(-1.0, 1.0)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("[GPT]hexagon_phase_function.png")
    plt.show()


if __name__ == "__main__":
    result = simulate(
        a=1.0,
        n_ice=1.31,
        n_rays=10000,
        dtheta_deg=1.0,
        max_hits=12,
        cutoff=1e-7,
        scatter_bin_deg=1.0,
    )
    plot_results(result)
