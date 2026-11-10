# Chevron rack (herringbone teeth) with a variable-radius gear, CT / CF   version 8   2026-11-10
# Blender add-on, single file, needs Blender 4.5+ (numpy and bpy only). 1 Blender unit = 1 mm.
# Input: the force (or torque) at the start and at the end of the stroke and the angle the gear turns. The gear radius follows from the
# spring law, the teeth are involute, the rack is lapped from the gear, the gear is lapped from that rack. Every layer of the face width
# is such a pair with the teeth shifted in phase (mirror image in the second half = chevron); the layers are lofted into closed solids.
bl_info = {
    "name": "Chevron rack CT / CF (variable radius)",
    "author": "darkfrei",
    "version": (8, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Chevron rack",
    "description": "Chevron rack with a variable-radius gear: constant-torque / constant-force pairs from the force at the ends of the stroke",
    "category": "Add Mesh",
}

import math
import numpy as np

# ------------------------------------------------------------------------------------------------ kinematics

class Kin:
    """case 1 (CT): r = r0 / sqrt(1 + 2 kappa t), kappa = k r0 / F0, force on the rack F0 + k s, torque on the gear F0 r0.
    case 2 (CF): r = r0 + c t, c = k_theta / F, r0 = tau0 / F, force F, torque tau0 + k_theta t (k_theta in N*mm/rad)"""
    def __init__(self, case, **p):
        self.case = int(case)
        if self.case == 1:
            self.F0, self.k, self.r0 = float(p['F0']), float(p['k']), float(p['r0'])
            if self.F0 <= 0 or self.r0 <= 0 or self.k < 0:
                raise ValueError("CT: the preload force and the start radius must be above 0")
            self.kappa = self.k * self.r0 / self.F0
            self.T = self.F0 * self.r0
        else:
            self.F, self.tau0, self.kth = float(p['F']), float(p['tau0']), float(p['kth'])
            if self.F <= 0 or self.tau0 <= 0 or self.kth < 0:
                raise ValueError("CF: the rack force and the preload torque must be above 0")
            self.r0 = self.tau0 / self.F
            self.c = self.kth / self.F

    def r(self, t):
        t = np.asarray(t, float)
        if self.case == 1:
            return self.r0 / np.sqrt(1 + 2 * self.kappa * t) if self.kappa > 1e-12 else self.r0 + 0 * t
        return self.r0 + self.c * t

    def rp(self, t):
        t = np.asarray(t, float)
        if self.case == 1:
            return -self.kappa * self.r0 * (1 + 2 * self.kappa * t) ** -1.5 if self.kappa > 1e-12 else 0 * t
        return self.c + 0 * t

    def s(self, t):
        t = np.asarray(t, float)
        if self.case == 1:
            if self.kappa < 1e-12:
                return self.r0 * t
            return self.r0 / self.kappa * (np.sqrt(1 + 2 * self.kappa * t) - 1)
        return self.r0 * t + 0.5 * self.c * t * t

    def tOfS(self, s):
        s = np.asarray(s, float)
        if self.case == 1:
            if self.kappa < 1e-12:
                return s / self.r0
            return ((self.kappa * s / self.r0 + 1) ** 2 - 1) / (2 * self.kappa)
        if abs(self.c) < 1e-12:
            return s / self.r0
        return (-self.r0 + np.sqrt(np.maximum(self.r0 ** 2 + 2 * self.c * s, 0.0))) / self.c

    def tilt(self, t):
        return np.arctan(-self.rp(t) / self.r(t))

    def load(self, t):
        t = np.asarray(t, float)
        if self.case == 1:
            return self.F0 + self.k * self.s(t), self.T + 0 * t
        return self.F + 0 * t, self.tau0 + self.kth * t

class KinShift:
    """the same kinematics with the turn counted from -e: lets the layers be built a bit beyond both ends of the usage angle"""
    def __init__(self, kin, e):
        self.k, self.e, self.case = kin, float(e), kin.case
        self.s0 = float(kin.s(-self.e))
    def r(self, t):
        return self.k.r(np.asarray(t, float) - self.e)
    def rp(self, t):
        return self.k.rp(np.asarray(t, float) - self.e)
    def s(self, t):
        return self.k.s(np.asarray(t, float) - self.e) - self.s0
    def tOfS(self, s):
        return self.k.tOfS(np.asarray(s, float) + self.s0) + self.e
    def tilt(self, t):
        return self.k.tilt(np.asarray(t, float) - self.e)
    def load(self, t):
        return self.k.load(np.asarray(t, float) - self.e)

# ------------------------------------------------------------------------------------------------ geometry helpers

def inv_theta(r, a, R):
    """polar angle of the involute flank of the base circle r cos(a), relative to its crossing of the pitch circle r (a in radians > 0)"""
    rb = r * math.cos(a)
    R = np.asarray(R, float)
    b = np.arccos(np.clip(rb / np.maximum(R, 1e-9), -1, 1))
    return np.where(R >= rb, np.tan(b) - b, 0.0) - (math.tan(a) - a)

def resample(P, h):
    P = np.asarray(P, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] < 1e-12:
        return P[:1]
    m = int(cum[-1] / h) + 2
    u = np.linspace(0.0, cum[-1], m)
    return np.stack([np.interp(u, cum, P[:, 0]), np.interp(u, cum, P[:, 1])], 1)

def dp_keep(P, tol):
    P = np.asarray(P, float)
    n = len(P)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        A, B = P[i0], P[i1]
        d = B - A
        L = math.hypot(*d)
        seg = P[i0 + 1:i1]
        dist = np.hypot(*(seg - A).T) if L < 1e-15 else np.abs((seg[:, 0] - A[0]) * d[1] - (seg[:, 1] - A[1]) * d[0]) / L
        j = int(dist.argmax())
        if dist[j] > tol:
            k = i0 + 1 + j
            keep[k] = True
            stack.append((i0, k)); stack.append((k, i1))
    return np.nonzero(keep)[0]

def to_rack(G, t, sv):
    a = -math.pi / 2 - t
    c, s_ = math.cos(a), math.sin(a)
    return np.stack([sv + c * G[..., 0] - s_ * G[..., 1], s_ * G[..., 0] + c * G[..., 1]], axis=-1)

# ------------------------------------------------------------------------------------------------ design

def build_design(kin, dth, n, delta=3.0, a_min=1.0, aw_const=None, ab=30.0, add=1.0, ded=1.25, thr=0.5, side='R', phase=0.0, law=0.0):
    """n gear teeth on the working arc (n + 1 rack teeth). working flank angle aw_i = max(a_min, tilt_i + delta), or aw_const.
    phase 0..1: all tooth positions shifted by this fraction of the angular pitch (a slice of a helical tooth), shapes follow the local radius"""
    # tooth index as a function of the turn: law 0 = equal angle (rack pitch ~ r), 1 = equal rack pitch, 2 = rack pitch ~ 1/r (CT: ~ force)
    tt = np.linspace(0.0, dth, 4001); ww = np.asarray(kin.r(tt)) ** (1.0 - law); XX = np.asarray(kin.s(tt))
    dk = np.r_[0.0, np.cumsum(np.diff(XX) / (0.5 * (ww[1:] + ww[:-1])))]; dk *= n / dk[-1]
    idx = np.arange(n + 1, dtype=float) if phase == 0.0 else np.arange(-1, n + 1) + phase
    t = np.interp(idx, dk, tt)
    s0 = (dk[1] - dk[0]) / (tt[1] - tt[0]); s1 = (dk[-1] - dk[-2]) / (tt[-1] - tt[-2])
    t = np.where(idx < 0, idx / s0, np.where(idx > n, dth + (idx - n) / s1, t))
    n1 = len(t) - 1
    X = np.asarray(kin.s(t)); Y = -np.asarray(kin.r(t))
    d = np.diff(X)
    p = np.empty(n1 + 1); p[0], p[-1] = d[0], d[-1]; p[1:-1] = 0.5 * (d[:-1] + d[1:])
    m = p / math.pi
    psi = np.degrees(kin.tilt(t))
    aw = np.full(n1 + 1, float(aw_const)) if aw_const is not None else np.clip((psi if side == 'R' else -psi) + delta, a_min, 60.0)
    return dict(ktt=tt, kdk=dk, side=side, kin=kin, t=t, X=X, Y=Y, p=p, m=m, psi=psi, aw=aw, ab=float(ab), th=thr * p, ha=add * m, ag=add * m, ded=ded * m,
                dth=float(dth), n=int(n), delta=float(delta), add=float(add), ded_f=float(ded), thr=float(thr))

def gear_outline(des, nflank=160, droot=0.002):
    """outer outline of the sector of the involute gear, phi from 0 to dth: (phi, R) vertices, phi non-decreasing"""
    kin, t, X, p = des['kin'], des['t'], des['X'], des['p']
    n1 = len(t)
    def root_at(ph):
        ph = np.asarray(ph, float)
        return kin.r(ph) - np.interp(kin.s(ph), X, des['ded'])
    last = 0.0
    out_phi = [0.0]; out_R = [float(root_at(0.0))]
    for i in range(n1):
        r = float(kin.r(t[i])); ded = float(des['ded'][i]); ag = float(des['ag'][i])
        Rr, Rt = r - ded, r + ag
        R = np.linspace(Rr, Rt, nflank)
        th0 = t[i] + (des['th'][i] / 2) / r
        w = (p[i] - des['th'][i]) / r
        if des.get('side', 'R') == 'R':          # the load is on the right flank of the rack tooth = the start side of the gear tooth
            thw = th0 + inv_theta(r, math.radians(des['aw'][i]), R)
            thb = th0 + w - inv_theta(r, math.radians(des['ab']), R)
        else:                                      # the load is on the left flank = the end side of the gear tooth
            thw = th0 + inv_theta(r, math.radians(des['ab']), R)
            thb = th0 + w - inv_theta(r, math.radians(des['aw'][i]), R)
        if thw[0] > last + droot:
            g = np.arange(last, thw[0], droot)
            out_phi += list(g); out_R += list(root_at(g))
        out_phi += list(np.maximum(thw, last)); out_R += list(R)
        tip = np.linspace(thw[-1], max(thb[-1], thw[-1]), 6)
        out_phi += list(tip); out_R += [Rt] * len(tip)
        out_phi += list(thb[::-1]); out_R += list(R[::-1])
        last = max(last, float(thb[0]))
    g = np.arange(last, des['dth'] + 1e-9, droot)
    out_phi += list(g); out_R += list(root_at(g))
    ph = np.maximum.accumulate(np.array(out_phi)); R = np.array(out_R)
    # clip to [0, dth]
    keep = (ph >= 0) & (ph <= des['dth'])
    ph, R = ph[keep], R[keep]
    # blank (tip) radius never exceeded
    Rtop = kin.r(ph) + np.interp(kin.s(ph), X, des['ag'])
    R = np.minimum(R, Rtop)
    return ph, R

def blank_top(des, margin=2.0):
    """rack blank: flat tips at their own height; returns (xs steps, ytop for every x via function)"""
    X = des['X']; Y = des['Y'] + des['ha']
    xm = np.r_[X[0] - margin, 0.5 * (X[:-1] + X[1:]), X[-1] + margin]
    def top(x):
        i = np.clip(np.searchsorted(xm, x, side='right') - 1, 0, len(X) - 1)
        return Y[i]
    return xm, top

# ------------------------------------------------------------------------------------------------ lapping

def lap_rack(des, ph0, R0, h=0.002, dx=0.002, step=0.04, margin=2.0, zoned=False):
    """rack = blank minus the sweep of the gear outline (height field). returns the rack top polyline (x, y)"""
    kin = des['kin']; t_end = des['dth']
    P0 = np.stack([R0 * np.cos(ph0), R0 * np.sin(ph0)], 1)
    P = resample(P0, h)
    phi = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    Rp = np.hypot(P[:, 0], P[:, 1])
    head = Rp > kin.r(phi)                       # gear head = outside its pitch curve (zoned lapping: it may cut only below the rack pitch curve)
    X = des['X']
    xm, top = blank_top(des, margin)
    x0, x1 = xm[0], xm[-1]
    nb = int(round((x1 - x0) / dx)) + 1
    xc = x0 + dx * np.arange(nb)
    f = top(xc).astype(float)
    ymax = float((des['Y'] + des['ha']).max()) + 0.01
    nst = int(math.ceil(max(t_end * (max(kin.r(0.0), kin.r(t_end)) + float(des['ag'].max())) / step, des['X'][-1] / step))) + 1
    ts = np.linspace(0.0, t_end, nst)
    wph = 2.8 * float(np.max(des['p'])) / float(min(kin.r(0.0), kin.r(t_end)))
    Px = np.ascontiguousarray(P[:, 0]); Py = np.ascontiguousarray(P[:, 1])
    svs = np.asarray(kin.s(ts), float) * np.ones(nst)
    I0 = np.searchsorted(phi, ts - wph); I1 = np.searchsorted(phi, ts + wph)
    for j in range(nst):
        i0, i1 = I0[j], I1[j]
        if i1 <= i0:
            continue
        a = -math.pi / 2 - ts[j]
        c, s_ = math.cos(a), math.sin(a)
        px = Px[i0:i1]; py = Py[i0:i1]
        qx = svs[j] + c * px - s_ * py; qy = s_ * px + c * py            # the gear outline in the rack frame
        ok = qy < ymax
        if zoned:
            Yp = -kin.r(kin.tOfS(np.clip(qx, 0.0, X[-1])))
            ok &= np.where(head[i0:i1], qy <= Yp, qy >= Yp)
        idx = np.rint((qx - x0) / dx).astype(np.int64)
        ok &= (idx >= 0) & (idx < nb)
        np.minimum.at(f, idx[ok], qy[ok])
    # holes inside the swept range: not touched bins keep the blank height. Where both neighbours are lower, take the lower
    # one of the two (a gap of one bin or two cannot be a real feature)
    return xc, f

def rack_polyline(xc, f, tol=0.0015):
    P = np.stack([xc, f], 1)
    keep = dp_keep(P, tol)
    return P[keep]

def lap_gear(des, Prack, h=0.002, dphi=1.0e-4, step=0.04, wfac=2.8, zoned=False):
    """gear = blank minus the sweep of the rack (polar bins). returns (phi, R) of the new outline"""
    kin = des['kin']; t_end = des['dth']
    P = resample(Prack, h)
    rhead = P[:, 1] >= -kin.r(kin.tOfS(np.clip(P[:, 0], 0.0, des['X'][-1])))     # rack head = above its pitch curve
    nb = int(math.ceil(t_end / dphi)) + 1
    phg = dphi * np.arange(nb)
    Rblank = kin.r(phg) + np.interp(kin.s(phg), des['X'], des['ag'])
    Rmin = np.full(nb, np.inf)
    nst = int(math.ceil(max(t_end * (max(kin.r(0.0), kin.r(t_end)) + float(des['ag'].max())) / step, des['X'][-1] / step))) + 1
    ts = np.linspace(0.0, t_end, nst)
    W = wfac * float(np.max(des['p']))
    xs = P[:, 0]
    Rcap = float(Rblank.max()) + 0.01
    Px = np.ascontiguousarray(P[:, 0]); Py = np.ascontiguousarray(P[:, 1])
    svs = np.asarray(kin.s(ts), float) * np.ones(nst)
    if np.all(np.diff(xs) >= 0):                     # the rack runs left to right: only the part near the gear is swept
        I0 = np.searchsorted(xs, svs - W); I1 = np.searchsorted(xs, svs + W)
    else:
        I0 = np.zeros(nst, int); I1 = np.full(nst, len(xs))
    for j in range(nst):
        i0, i1 = I0[j], I1[j]; tj = ts[j]
        a = math.pi / 2 + tj
        c, s_ = math.cos(a), math.sin(a)
        vx = Px[i0:i1] - svs[j]; vy = Py[i0:i1]
        qx = c * vx - s_ * vy; qy = s_ * vx + c * vy                    # the rack in the gear frame
        R = np.hypot(qx, qy)
        ok = R < Rcap
        ph = np.arctan2(qy, qx)
        ph = ph + 2 * math.pi * np.rint((tj - ph) / (2 * math.pi))
        if zoned:                                    # rack head cuts only the gear foot, rack foot only the gear head
            rp = kin.r(np.clip(ph, 0.0, t_end))
            ok &= np.where(rhead[i0:i1], R <= rp, R >= rp)
        idx = np.floor(ph / dphi).astype(np.int64)
        ok &= (idx >= 0) & (idx < nb)
        np.minimum.at(Rmin, idx[ok], R[ok])
    R2 = np.minimum(Rblank, Rmin)
    return phg, R2

def gear_polyline(phg, R2, tol=0.0015):
    P = np.stack([R2 * np.cos(phg), R2 * np.sin(phg)], 1)
    keep = dp_keep(P, tol)
    return P[keep]

# ------------------------------------------------------------------------------------------------ pair check

def pair_check(des, Prack, Pgear, npos=60, tol=0.004, mu_t=0.2, mu_g=0.2, h=0.01, conj=0.12, minlen=0.03):
    """walk the lapped pair along the stroke; at every place find where the gear outline touches the rack (distance < tol),
    group the touching samples into contact lines, keep those whose normal runs through the pole (conjugate),
    then the friction-consistent loss with equal sharing of the load"""
    kin = des['kin']
    R = resample(Prack, 0.002)
    G = resample(Pgear, h)
    X = des['X']
    ts = np.linspace(des['t'][3] if len(des['t']) > 8 else 0.0, des['t'][-4] if len(des['t']) > 8 else des['dth'], npos)
    rec = []
    pen = 0.0
    # rack normals from the polyline (outward = up)
    Tg = np.gradient(R, axis=0)
    ln = np.hypot(Tg[:, 0], Tg[:, 1]); ln[ln == 0] = 1
    Nr = np.stack([-Tg[:, 1], Tg[:, 0]], 1) / ln[:, None]
    Nr[Nr[:, 1] < 0] *= -1                                 # upward
    awx = lambda x: np.interp(x, X, des['aw'])
    W = 2.6 * float(np.max(des['p']))
    for tj in ts:
        sv = float(kin.s(tj)); rj = float(kin.r(tj))
        Q = to_rack(G, tj, sv)
        near = np.abs(Q[:, 0] - sv) < W
        Qn = Q[near]
        if len(Qn) == 0:
            rec.append(dict(t=float(tj), s=sv, r=rj, cl=[])); continue
        # prefilter by vertical gap
        fy = np.interp(Qn[:, 0], R[:, 0], R[:, 1])
        cand = (Qn[:, 1] - fy) < 0.08
        cl = []
        if cand.any():
            idxn = np.nonzero(near)[0][cand]
            Qc = Qn[cand]
            c0 = np.clip(np.searchsorted(R[:, 0], Qc[:, 0]), 0, len(R) - 1)
            off = np.arange(-70, 71)
            J = np.clip(c0[:, None] + off[None, :], 0, len(R) - 1)
            D = ((Qc[:, None, :] - R[J]) ** 2).sum(2)
            ja = D.argmin(1)
            d2 = D[np.arange(len(Qc)), ja]; jn = J[np.arange(len(Qc)), ja]
            Nm = Nr
            inside = Qc[:, 1] < fy[cand]
            if inside.any():
                pen = max(pen, float(np.sqrt(d2[inside].max())))
            okk = d2 < tol * tol
            if okk.any():
                ii = idxn[okk]; Qk = Qc[okk]; nk = Nm[jn[okk]]
                # side: the normal angle near the flank angle at this place
                ang = np.degrees(np.arctan2(nk[:, 1], nk[:, 0]))
                tgt = awx(Qk[:, 0]) if des.get('side', 'R') == 'R' else 180.0 - awx(Qk[:, 0])
                da = (ang - tgt + 180.0) % 360.0 - 180.0
                side = (da > -25) & (da < 35) if des.get('side', 'R') == 'R' else (da > -35) & (da < 25)
                ii, Qk, nk = ii[side], Qk[side], nk[side]
                if len(ii):
                    brk = np.nonzero(np.diff(ii) > 8)[0] + 1
                    for seg in np.split(np.arange(len(ii)), brk):
                        L = h * len(seg)
                        if L < minlen:
                            continue
                        c = Qk[seg].mean(0); nm = nk[seg].mean(0); nm = nm / max(float(np.hypot(*nm)), 1e-12)
                        v = c - np.array([sv, -rj]); lv = float(np.hypot(*v))
                        sn = abs(nm[0] * v[1] - nm[1] * v[0]) / max(lv, 1e-9)
                        if sn < conj:
                            cl.append(dict(c=c, n=nm, L=L))
        rec.append(dict(t=float(tj), s=sv, r=rj, cl=cl))
    Wt = []; mesh = []; guide = []; tot = []; m = []; Ls = []; Fs = []; dead = 0
    for r in rec:
        cl = r['cl']
        if not cl:
            dead += 1
            continue
        pole = np.array([r['s'], -r['r']])
        n = np.array([c['n'] for c in cl]); C = np.array([c['c'] for c in cl])
        d = ((C - pole) * n).sum(1)
        s = np.sign(-d)
        A = n[:, 0] + mu_t * s * n[:, 1]
        B = -n[:, 1] + mu_t * s * n[:, 0]
        sd = np.abs(d).sum() / r['r']
        if des.get('side', 'R') == 'R':          # the gear drives the rack (CT): the guide friction is part of the load
            den = A.sum() - mu_g * abs(B.sum())
        else:                                      # the rack drives the gear (CF): the tooth friction takes torque
            den = (-A).sum() - mu_t * sd
        if den < 0.05:
            den = 0.05
        Nn = 1.0 / den
        me = mu_t * Nn * sd
        gu = mu_g * Nn * abs(B.sum())
        mesh.append(me); guide.append(gu); tot.append((me + gu) / (1 + me + gu)); m.append(len(cl))
        Wt.append(float(kin.load(r['t'])[1])); Ls.append(sum(c['L'] for c in cl)); Fs.append(float(kin.load(r['t'])[0]))
    out = dict(contact=1.0 - dead / len(rec), positions=len(rec), rec=rec, pen=pen)
    if Wt:
        Wt = np.array(Wt); Wt /= Wt.sum()
        f = lambda v: float((np.array(v) * Wt).sum())
        out.update(mesh=f(mesh), guide=f(guide), loss=f(tot), touches=float(np.mean(m)), sumL=float(np.mean(Ls)),
                   force_per_len=float(np.mean(np.array(Fs) / np.maximum(Ls, 1e-3))),
                   fl_p95=float(np.percentile(np.array(Fs) / np.maximum(Ls, 1e-3), 95)))
    else:
        out.update(mesh=float('nan'), guide=float('nan'), loss=float('nan'), touches=0.0, sumL=0.0, force_per_len=float('nan'), fl_p95=float('nan'))
    return out

def rack_clearance(P, c, steep=0.5, side='R'):
    """print clearance: everything of the rack except the working flank (right flank of every tooth) goes down by c:
    tooth tips (root clearance of the gear), the rack floor (tip clearance of the gear) and the back flank (back lash on the side
    that carries no load). the working flank keeps the lapped shape, so the load is carried exactly as checked."""
    P = np.asarray(P, float).copy()
    if c <= 0:
        return P
    d = np.diff(P, axis=0)
    work = ((d[:, 1] < 0) if side == 'R' else (d[:, 1] > 0)) & (np.abs(d[:, 1]) > steep * np.abs(d[:, 0]))   # steep flank of the loaded side
    seg_w = np.r_[False, work] | np.r_[work, False]                                  # vertices touching a working flank segment
    P[~seg_w, 1] -= c
    return P

def morph(f, r, dx, mode):
    """height field f on a grid of step dx, material below f. 'open': what a circle of radius r can draw from inside
    (tips get radius >= r, thinner parts go), 'close': what it can draw from outside (roots get radius >= r)"""
    k = int(round(r / dx))
    if k < 1:
        return f
    s = np.sqrt(np.maximum(r * r - (np.arange(-k, k + 1) * dx) ** 2, 0.0))
    g = np.r_[np.full(k, f[0]), f, np.full(k, f[-1])]
    def ero(a):
        o = np.full_like(a, np.inf)
        for j, du in enumerate(range(-k, k + 1)):
            o = np.minimum(o, np.roll(a, -du) - s[j])
        return o
    def dil(a):
        o = np.full_like(a, -np.inf)
        for j, du in enumerate(range(-k, k + 1)):
            o = np.maximum(o, np.roll(a, -du) + s[j])
        return o
    g = dil(ero(g)) if mode == 'open' else ero(dil(g))
    return g[k:-k]

def make_pair_round(des, quality='fine', clearance=0.0, rnd=0.2):
    """the printable pair: the rack is what a circle of radius rnd can draw (all corners >= rnd), the gear is lapped from it and
    rounded the same way (its tips), the print clearance is put on the rack and the rack is rounded again"""
    if quality == 'fine':
        h, dx, dphi, step, tol = 0.002, 0.002, 1.0e-4, 0.04, 0.0015
    else:
        h, dx, dphi, step, tol = 0.004, 0.004, 2.0e-4, 0.07, 0.003
    kin = des['kin']; side = des.get('side', 'R')
    rr = lambda a: morph(morph(a, rnd, dx, 'close'), rnd, dx, 'open') if rnd > 0 else a
    ph0, R0 = gear_outline(des)
    xc, f = lap_rack(des, ph0, R0, h=h, dx=dx, step=step)
    f = rr(f)
    Pr = rack_polyline(xc, f, tol)
    phg, R2 = lap_gear(des, Pr, h=h, dphi=dphi, step=step)
    if rnd > 0:
        Rt = np.asarray(kin.r(phg)) + np.interp(np.asarray(kin.s(phg)), des['X'], des['ag'])
        u = np.r_[0.0, np.cumsum(0.5 * (Rt[1:] + Rt[:-1]) * np.diff(phg))]       # arc length near the tips
        ug = np.arange(0.0, u[-1], dx)
        R2 = np.interp(u, ug, rr(np.interp(ug, u, R2)))
    Pg = gear_polyline(phg, R2, tol)
    Pc = rack_clearance(Pr, clearance, side=side)
    fc = rr(np.interp(xc, Pc[:, 0], Pc[:, 1]))
    return dict(rack=rack_polyline(xc, fc, tol), gear=Pg)

def chevron_eval(ck, des, frac=1.0):
    """each half of the chevron shifts the teeth by a whole pitch, so at any stroke position the layers see every relative position
    within one pitch: the contact length of the pair is the 2D contact length averaged over one pitch. dead = share of the stroke
    where no layer touches"""
    s = np.array([r['s'] for r in ck['rec']]); L = np.array([sum(c['L'] for c in r['cl']) for r in ck['rec']])
    p = np.interp(s, des['X'], des['p'])
    frac = min(1.0, max(frac, 1e-6))                      # share of a pitch the layers of one half cover
    eff = np.array([L[np.abs(s - si) <= 0.5 * frac * pi_ + 1e-9].mean() for si, pi_ in zip(s, p)])
    return eff, float((eff < 0.02).mean())

def tip_widths(prof_x, prof_y, depth=0.1):
    """widths of the peaks of a single valued profile at 'depth' below each top"""
    out = []
    y = prof_y
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            lv = y[i] - depth; l = i
            while l > 0 and y[l] > lv: l -= 1
            q = i
            while q < len(y) - 1 and y[q] > lv: q += 1
            if l > 0 and q < len(y) - 1 and (y[i] - min(y[max(0, l - 400):l + 1].min(), y[q:q + 400].min())) > 0.3:
                out.append(prof_x[q] - prof_x[l])
    return np.array(out)

def pair_tips(des, pr):
    R = pr['rack']; x = np.arange(R[0, 0], R[-1, 0], 0.002); y = np.interp(x, R[:, 0], R[:, 1])
    X = des['X']; mk = (x > X[1]) & (x < X[-2])
    wr = tip_widths(x[mk], y[mk])
    G = pr['gear']; ph = np.maximum.accumulate(np.unwrap(np.arctan2(G[:, 1], G[:, 0]))); rr_ = np.hypot(G[:, 0], G[:, 1])
    pg = np.arange(ph[0], ph[-1], 1e-4); Rg = np.interp(pg, ph, rr_)
    t = des['t']; mg = (pg > t[1]) & (pg < t[-2])
    wg = tip_widths(pg[mg] * Rg[mg], Rg[mg])
    return wr, wg

# ------------------------------------------------------------------------------------------------ blender part
import bpy
import bmesh
from bpy.props import (EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty)
from mathutils import Vector
from mathutils import geometry as mgeo

POSE_TARGET = {}                 # scene name -> (rack name, gear name, kinematics)

def _pose_update(self, context):
    apply_pose(context.scene)

class ChevProps(bpy.types.PropertyGroup):
    case: EnumProperty(name="Pair", items=[
        ('CT', "CT: constant torque", "the spring pushes the rack, the torque on the gear stays the same, the gear radius shrinks"),
        ('CF', "CF: constant force", "the torsion spring turns the gear, the force on the rack stays the same, the gear radius grows")],
        default='CF')
    # ---- the working range: where the spring is at the start and at the end of the stroke, and which part of the gear turns
    r_start: FloatProperty(name="Gear radius at start (mm)", default=15.0, min=2.0, soft_max=200.0)
    f_start: FloatProperty(name="Force at start (N)", description="CT: spring force at the start (preload); CF: the constant force on the rack",
                           default=20.0, min=0.01)
    ratio: FloatProperty(name="End / start", description="CT: spring force at the end / at the start = gear radius at the start / at the end; "
                         "CF: spring torque at the end / at the start = gear radius at the end / at the start", default=1.4, min=1.01, max=6.0)
    use_angle: FloatProperty(name="Gear usage angle (deg)", description="how far the gear turns over the whole stroke; the toothed sector of the gear is this big",
                             default=160.0, min=20.0, max=340.0)
    start_angle: FloatProperty(name="Sector start angle (deg)", description="turns the toothed sector round the gear axle (the position of the sector on the gear body)",
                               default=0.0, soft_min=-180.0, soft_max=180.0)
    # ---- spring helper (optional)
    sp_d: FloatProperty(name="Wire diameter (mm)", description="0 = not given", default=0.0, min=0.0, precision=3)
    sp_D: FloatProperty(name="Mean coil diameter (mm)", default=0.0, min=0.0)
    sp_mod: FloatProperty(name="G or E (MPa)", description="compression / extension: shear modulus G (steel 79300); torsion: Young's modulus E (steel 200000)",
                          default=0.0, min=0.0)
    sp_allow: FloatProperty(name="Allowed stress (MPa)", default=700.0, min=1.0)
    # ---- teeth
    m_min: FloatProperty(name="Smallest module (mm)", description="smallest module on the rack; sets the number of teeth", default=1.2, min=0.2, soft_max=6.0)
    add: FloatProperty(name="Tooth height (modules)", default=1.0, min=0.4, max=1.6)
    law: EnumProperty(name="Pitch law", items=[
        ('AUTO', "Auto", "CT: the module grows with the force (the loaded teeth are the big ones); CF: equal angle on the gear"),
        ('ANGLE', "Equal gear angle", "rack pitch follows the gear radius"),
        ('EQUAL', "Equal rack pitch", "one module along the whole rack"),
        ('FORCE', "Module grows with the force", "CT: rack pitch ~ 1/radius ~ spring force")], default='AUTO')
    thr_mode: EnumProperty(name="Tooth thickness", items=[
        ('AUTO', "Auto", "the rack tooth / pitch share is searched together with the flank angle (the gear is open, any share works)"),
        ('FIXED', "Fixed", "use the share below")], default='AUTO')
    thr: FloatProperty(name="Rack tooth thickness / pitch", default=0.5, min=0.3, max=0.7)
    nozzle: FloatProperty(name="Nozzle diameter (mm)", description="every corner of both parts gets at least half of this as radius (what this nozzle can draw); 0 = sharp",
                          default=0.4, min=0.0, max=1.2)
    angle_mode: EnumProperty(name="Flank angle", items=[
        ('AUTO', "Auto", "CT: follows the tilt of the pitch curve, offset searched; CF: constant angle searched (5..13 deg)"),
        ('OFFSET', "Tilt + offset", "CT only: angle = tilt of the pitch curve + the offset"),
        ('CONST', "Constant", "one pressure angle for all teeth")], default='AUTO')
    offset: FloatProperty(name="Offset (deg)", default=6.0, soft_min=0.0, soft_max=15.0)
    alpha: FloatProperty(name="Pressure angle (deg)", default=9.0, min=1.0, max=45.0)
    back: FloatProperty(name="Back flank angle (deg)", default=30.0, min=1.0, max=45.0)
    # ---- chevron
    helix: FloatProperty(name="Helix angle (deg)", description="tooth slope at the reference radius; the two halves of the face width are mirror images (chevron)",
                         default=30.0, min=5.0, max=60.0)
    helix_at: EnumProperty(name="Helix angle at", items=[('MIN', "smallest radius", ""), ('MEAN', "mean radius", ""), ('MAX', "largest radius", "")], default='MEAN')
    layers: IntProperty(name="Layers per half", description="layers over each half of the face width; the vertices follow the teeth, so few are needed",
                        default=6, min=2, max=60)
    min_edge: FloatProperty(name="Edge length (mm)", description="the profile is sampled along the teeth with this step, so no edge is shorter than this "
                            "(a 3D printer cannot print finer: the nozzle line is 0.4 mm wide, a resin pixel 0.02-0.05 mm); the thinning below only makes edges longer",
                            default=0.05, min=0.01, max=1.0, precision=3)
    tol: FloatProperty(name="Mesh tolerance (mm)", description="greedy thinning of the edges along the teeth: the edge that changes the volume least goes "
                       "first, as long as the surface stays within this distance of the fine profile in every layer", default=0.005, min=0.0005, max=0.1, precision=4)
    width: FloatProperty(name="Face width (mm)", description="whole width of both halves together", default=10.0, min=1.0)
    base: FloatProperty(name="Rack base thickness (mm)", default=4.0, min=0.5)
    bore: FloatProperty(name="Shaft bore (mm)", default=6.0, min=0.0)
    clear: FloatProperty(name="Print clearance (mm)", default=0.2, min=0.0, max=1.0)
    mu_t: FloatProperty(name="Tooth friction", default=0.2, min=0.0, max=1.0)
    mu_g: FloatProperty(name="Rack guide friction", default=0.2, min=0.0, max=1.0)
    quality: EnumProperty(name="Quality", items=[('FINE', "Fine (2 um steps)", ""), ('COARSE', "Draft (4 um steps)", "")], default='COARSE')
    pose: FloatProperty(name="Stroke", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_pose_update)
    report: StringProperty(default="")

# --- the working range -> kinematics ---------------------------------------------------------------------------------

def plan_ends(P, m_min=None):
    """start/end of the stroke + gear usage angle -> kinematics, spring numbers, number of teeth"""
    m_min = P.m_min if m_min is None else m_min
    out = dict(ok=True, err=[], warn=[])
    dth = math.radians(P.use_angle); rho = P.ratio
    try:
        if P.case == 'CT':
            F1, r1 = P.f_start, P.r_start
            F2 = rho * F1
            T = F1 * r1
            k = (F2 ** 2 - F1 ** 2) / (2.0 * T * dth)                     # F^2 = F1^2 + 2 k T t
            kin = Kin(1, F0=F1, k=k, r0=r1)
            S = (F2 - F1) / k
            out.update(F1=F1, F2=F2, k=k, T=T, S=S, r_end=r1 / rho)
        else:
            F, r1 = P.f_start, P.r_start
            r2 = rho * r1
            tau1, tau2 = F * r1, F * r2
            kth = (tau2 - tau1) / dth                                        # N*mm per rad
            kin = Kin(2, F=F, tau0=tau1, kth=kth)
            S = float(kin.s(dth))
            out.update(F=F, tau1=tau1, tau2=tau2, k=kth * math.pi / 180.0, S=S, r_end=r2, wind=P.use_angle)
    except ValueError as e:
        out['ok'] = False; out['err'].append(str(e)); return out
    tilt = np.degrees(np.abs(kin.tilt(np.linspace(0.0, dth, 50))))
    if tilt.max() > 45.0:
        out['ok'] = False
        out['err'].append("the pitch curve tilts up to %.0f deg (limit 45): lower the end/start ratio or widen the usage angle" % tilt.max())
        return out
    r_a, r_b = float(kin.r(0.0)), float(kin.r(dth))
    rmin = min(r_a, r_b)
    law = law_of(P)
    tt = np.linspace(0.0, dth, 2001); ww = np.asarray(kin.r(tt)) ** (1.0 - law); XX = np.asarray(kin.s(tt))
    cc = math.pi * m_min / ww.min()                                         # the smallest rack pitch = pi * smallest module
    n = max(4, int(round(float(np.sum(np.diff(XX) / (cc * 0.5 * (ww[1:] + ww[:-1])))))))
    out.update(law=law, kin=kin, dth=dth, S=S, n=n, r_start=r_a, r_end=r_b, rmin=rmin, rmax=max(r_a, r_b), m_min=m_min, tilt_max=float(tilt.max()))
    pmin = math.pi * m_min
    out['pmin'] = pmin
    root = rmin - (1.05 * P.add + 0.25) * pmin / math.pi
    if P.bore > 0 and root < P.bore / 2 + 2.0:
        out['warn'].append("the gear root radius %.1f mm leaves less than 2 mm of material around the bore" % root)
    if P.sp_d > 0 and P.sp_D > P.sp_d and P.sp_mod > 0:
        C = P.sp_D / P.sp_d
        if P.case == 'CT':
            n_c = P.sp_mod * P.sp_d ** 4 / (8 * P.sp_D ** 3 * out['k'])
            Kw = (4 * C - 1) / (4 * C - 4) + 0.615 / C
            tau = Kw * 8 * out['F2'] * P.sp_D / (math.pi * P.sp_d ** 3)
            out['spring'] = "active coils %.1f | shear at the end force %.0f MPa" % (n_c, tau)
            if tau > P.sp_allow:
                out['warn'].append("the spring shear %.0f MPa is above the allowed %.0f" % (tau, P.sp_allow))
        else:
            n_c = P.sp_mod * P.sp_d ** 4 * math.pi / (64 * P.sp_D * out['k'] * 180.0)
            Kb = (4 * C * C - C - 1) / (4 * C * (C - 1))
            sig = Kb * 32 * out['tau2'] / (math.pi * P.sp_d ** 3)
            out['spring'] = "active coils %.1f | bending at the end torque %.0f MPa" % (n_c, sig)
            if sig > P.sp_allow:
                out['warn'].append("the spring bending %.0f MPa is above the allowed %.0f" % (sig, P.sp_allow))
    return out

def law_of(P):
    if P.law == 'AUTO':
        return 2.0 if P.case == 'CT' else 0.0
    return {'ANGLE': 0.0, 'EQUAL': 1.0, 'FORCE': 2.0}[P.law]

def chevron_shift(P, pl):
    """pitches the teeth shift over each half of the face width"""
    r_ref = {'MIN': pl['rmin'], 'MEAN': 0.5 * (pl['r_start'] + pl['r_end']), 'MAX': pl['rmax']}[P.helix_at]
    return math.tan(math.radians(P.helix)) / r_ref * 0.5 * P.width / (pl['dth'] / pl['n'])

def search_design(P, pl, add, log):
    """flank angle x rack tooth share, judged as a chevron: no stroke place where no layer touches, no interference;
    score = loss (%) + 0.2 x contact pressure (MPa) at the weakest place. returns (delta, aw_const, thr) or None"""
    kin, dth, n, law = pl['kin'], pl['dth'], pl['n'], pl['law']
    side = 'R' if P.case == 'CT' else 'L'
    F = pl['F2'] if P.case == 'CT' else pl['F']
    if P.angle_mode == 'CONST':
        angs = [(None, P.alpha)]
    elif P.angle_mode == 'OFFSET':
        angs = [(P.offset, None)]
    else:
        angs = [(d, None) for d in (0, 4, 8, 12)] if P.case == 'CT' else [(None, a) for a in (5, 7, 9, 11)]
    thrs = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6) if P.thr_mode == 'AUTO' else (P.thr,)
    back, clear, rnd, mu_t, mu_g, width, shift = P.back, P.clear, 0.5 * P.nozzle, P.mu_t, P.mu_g, P.width, chevron_shift(P, pl)
    def one(c):
        d, a, th = c
        des = build_design(kin, dth, n, delta=(d if d is not None else 0.0), aw_const=a, ab=back, add=add, ded=add + 0.25,
                           thr=th, side=side, law=law)
        pr = make_pair_round(des, 'coarse', clear, rnd)
        ck = pair_check(des, pr['rack'], pr['gear'], npos=60, mu_t=mu_t, mu_g=mu_g)
        eff, dead = chevron_eval(ck, des, shift)
        ok = dead == 0.0 and ck['pen'] < 0.03
        return ok, 100 * ck['loss'] + 0.2 * F / max(float(eff.min()), 1e-3) / width
    cands = [(d, a, th) for d, a in angs for th in thrs]
    best = None
    for (d, a, th), (ok, score) in zip(cands, map(one, cands)):
        log.append(("%s/%.2f" % (("d%g" % d) if d is not None else ("a%g" % a), th), score if ok else None))
        if ok and (best is None or score < best[0]):
            best = (score, d, a, th)
    return None if best is None else best[1:]

# --- meshes ---------------------------------------------------------------------------------------------------

def decimate_greedy(layers, dz, tol):
    """greedy removal of whole longitudinal edges (one profile vertex in every layer = a line along the tooth).
    cost of removing a vertex = volume it adds or cuts away along the whole tooth (triangle prev-vertex-next in every layer x layer
    spacing); the cheapest goes first, but only while every original vertex the new edge spans stays within tol of it in every layer.
    returns the kept indices"""
    import heapq
    A = np.stack(layers, 0)                                            # (L, K, 2)
    K = A.shape[1]
    prv = np.arange(-1, K - 1); nxt = np.arange(1, K + 1); alive = np.ones(K, bool)
    def cost(k):
        p, q = prv[k], nxt[k]
        if p < 0 or q >= K:
            return None
        a, b, c = A[:, p], A[:, k], A[:, q]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        vol = float(area.sum()) * dz
        # deviation of every original vertex between p and q from the chord p-q, in every layer
        seg = A[:, p + 1:q]; d = c - a; ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
        dev = np.abs((seg[..., 0] - a[:, None, 0]) * d[:, None, 1] - (seg[..., 1] - a[:, None, 1]) * d[:, None, 0]) / ln[:, None]
        return vol, float(dev.max()) if dev.size else 0.0
    heap = []; ver = np.zeros(K, int)
    for k in range(1, K - 1):
        c_ = cost(k)
        if c_ and c_[1] <= tol:
            heapq.heappush(heap, (c_[0], k, 0))
    while heap:
        v, k, vv = heapq.heappop(heap)
        if not alive[k] or vv != ver[k]:
            continue
        c_ = cost(k)
        if c_ is None or c_[1] > tol:
            continue
        alive[k] = False
        p, q = prv[k], nxt[k]
        nxt[p] = q; prv[q] = p
        for j in (p, q):
            if 0 < j < K - 1 and alive[j]:
                ver[j] += 1
                c2 = cost(j)
                if c2 and c2[1] <= tol:
                    heapq.heappush(heap, (c2[0], j, ver[j]))
    return np.nonzero(alive)[0]

def loft_mesh(name, rows, zs):
    """closed solid from slices: rows[i] = [outer loop, hole loops...] (every row has the same number of vertices per loop),
    zs[i] = position of the slice along z"""
    bm = bmesh.new()
    nl = len(rows[0])
    flip = []
    for li in range(nl):
        a = 0.5 * float(np.sum(rows[0][li][:, 0] * np.roll(rows[0][li][:, 1], -1) - np.roll(rows[0][li][:, 0], -1) * rows[0][li][:, 1]))
        flip.append((a > 0) != (li == 0))                                  # outer counter-clockwise, holes clockwise
    V = []
    for i, row in enumerate(rows):
        vr = []
        for li, L in enumerate(row):
            L = L[::-1] if flip[li] else L
            vr.append([bm.verts.new((float(x), float(y), float(zs[i]))) for x, y in L])
        V.append(vr)
    for i in range(len(rows) - 1):
        for li in range(nl):
            a, b = V[i][li], V[i + 1][li]; m = len(a)
            for j in range(m):
                k = (j + 1) % m
                try:
                    bm.faces.new((a[j], a[k], b[k], b[j]))
                except ValueError:
                    pass
    def fill(loop):
        tris = mgeo.tessellate_polygon([[Vector((v.co.x, v.co.y, 0.0)) for v in loop]])
        for a, b, c in tris:
            try:
                bm.faces.new((loop[a], loop[b], loop[c]))
            except ValueError:
                pass
    for idx in (0, len(rows) - 1):                                     # the end caps
        rings = V[idx]
        if len(rings) == 1:
            fill(list(rings[0])); continue
        # one hole round the origin (the bore) and an outer ring that is star shaped round it (the gear): zip the two rings by angle
        o_ = list(rings[0]); h_ = list(rings[1])[::-1]                  # both counter-clockwise
        ao = np.unwrap([math.atan2(v.co.y, v.co.x) for v in o_])
        ah = np.unwrap([math.atan2(v.co.y, v.co.x) for v in h_])
        j0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (ah - ao[0]))))))
        h_ = h_[j0:] + h_[:j0]; ah = np.unwrap(np.r_[ah[j0:], ah[:j0]])
        ah = ah - 2 * math.pi * np.round((ah[0] - ao[0]) / (2 * math.pi))
        AO = np.r_[ao, ao[0] + 2 * math.pi]; AH = np.r_[ah, ah[0] + 2 * math.pi]
        no, nh = len(o_), len(h_)
        O_ = o_ + [o_[0]]; H_ = h_ + [h_[0]]
        i = j = 0
        while i < no or j < nh:
            if j >= nh or (i < no and AO[i + 1] <= AH[j + 1]):
                tri = (O_[i], O_[i + 1], H_[j]); i += 1
            else:
                tri = (O_[i], H_[j + 1], H_[j]); j += 1
            if len(set(tri)) == 3:
                try:
                    bm.faces.new(tri)
                except ValueError:
                    pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    return me

def apply_pose(scene):
    tgt = POSE_TARGET.get(scene.name)
    if not tgt:
        gears = [o for o in scene.objects if o.name.endswith("_gear") and "chev" in o]
        if not gears:
            return
        import ast
        g = sorted(gears, key=lambda o: o.name)[-1]
        tgt = (g.name.replace("_gear", "_rack"), g.name, ast.literal_eval(g["chev"]))
        POSE_TARGET[scene.name] = tgt
    _, gear, kj = tgt
    go = bpy.data.objects.get(gear)
    if go is None:
        return
    kin = Kin(**kj['kin'])
    t = float(scene.chev.pose) * kj['dth']
    go.location = (float(kin.s(t)), 0.0, 0.0)
    go.rotation_euler = (0.0, 0.0, -math.pi / 2 - t - kj.get('a0', 0.0))

class CHEV_OT_generate(bpy.types.Operator):
    bl_idname = "chev.generate"
    bl_label = "Generate the chevron rack pair"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import time
        P = context.scene.chev
        t0 = time.time()
        pl = plan_ends(P)
        if not pl['ok']:
            P.report = "ERROR\n" + "\n".join(pl['err'])
            self.report({'ERROR'}, pl['err'][0]); return {'CANCELLED'}
        auto = P.angle_mode == 'AUTO'
        mods = [P.m_min]
        while auto and mods[-1] * 0.85 >= 0.6:
            mods.append(mods[-1] * 0.85)
        adds = [P.add] + [a for a in ((1.0,) if P.case == 'CT' else (1.0, 1.2)) if auto and a > P.add]
        log = []; notes = []
        wm = context.window_manager
        wm.progress_begin(0, 100)
        side = 'R' if P.case == 'CT' else 'L'
        quality = 'fine' if P.quality == 'FINE' else 'coarse'
        try:
            done = None
            for mi, mm in enumerate(mods):
                plm = plan_ends(P, mm)
                if not plm['ok']:
                    break
                for add in adds:
                    found = search_design(P, plm, add, log)
                    wm.progress_update(min(40, 5 + 8 * mi))
                    if found is not None:
                        done = (plm, add) + tuple(found); break
                if done:
                    break
            if done is None:
                P.report = ("ERROR\nno flank angle and tooth thickness give a chevron without dead places, even with the smallest module %.2f mm.\n"
                            "Use more teeth (larger gear radius, smaller end/start ratio, wider usage angle), a smaller nozzle or set the angle by hand." % mods[-1])
                self.report({'ERROR'}, "no design without dead places; see the panel"); return {'CANCELLED'}
            pl, add_used, delta, awc, thr_used = done
            if abs(pl['m_min'] - P.m_min) > 1e-9:
                notes.append("smallest module lowered to %.2f mm (you asked %.2f): %d teeth" % (pl['m_min'], P.m_min, pl['n']))
            if abs(add_used - P.add) > 1e-9:
                notes.append("tooth height raised to %.1f modules (you asked %.1f) to get full contact" % (add_used, P.add))
            kin, dth, n = pl['kin'], pl['dth'], pl['n']
            pitch_ang = dth / n
            r_ref = {'MIN': pl['rmin'], 'MEAN': 0.5 * (pl['r_start'] + pl['r_end']), 'MAX': pl['rmax']}[P.helix_at]
            kappa = math.tan(math.radians(P.helix)) / r_ref                    # tooth phase turns by this angle per mm of face width
            bh = 0.5 * P.width
            shift = kappa * bh / pitch_ang                                    # pitches of shift over each half
            rnd = 0.5 * P.nozzle
            mkdes = lambda kk, dd, nn, q: build_design(kk, dd, nn, delta=(delta if delta is not None else 0.0), aw_const=awc, ab=P.back,
                                                      add=add_used, ded=add_used + 0.25, thr=thr_used, side=side, phase=q, law=pl['law'])
            # checks on three layers, as straight teeth (fine pairs)
            chk = []
            for j, q in enumerate(sorted(set((0.0, 0.5 * shift, shift)))):
                dq = mkdes(kin, dth, n, float(q % 1.0))
                pf = make_pair_round(dq, 'fine', P.clear, rnd)
                chk.append(pair_check(dq, pf['rack'], pf['gear'], npos=100, mu_t=P.mu_t, mu_g=P.mu_g))
                if j == 0:
                    eff, dead = chevron_eval(chk[-1], dq, shift); wr, wg = pair_tips(dq, pf); m_lo, m_hi = float(dq['m'][0]), float(dq['m'][-1])
                wm.progress_update(45 + 5 * j)
            # the layers are built a few teeth beyond both ends, so every layer has its teeth over the whole usage angle while the
            # vertices follow the teeth (quads along the tooth)
            t0s = mkdes(kin, dth, n, 0.0)['t']
            pa0, pa1 = float(t0s[1] - t0s[0]), float(t0s[-1] - t0s[-2])     # pitch angle at the start and at the end
            for E in range(int(math.ceil(shift)), int(math.ceil(shift)) + 4):
                e0, e1 = 1.1 * E * pa0, 1.1 * E * pa1
                if dth + e0 + e1 > 2 * math.pi - 0.3:
                    raise ValueError("the usage angle %.0f deg leaves no room for the chevron margins: lower the usage angle" % math.degrees(dth))
                if P.case == 'CT' and 1 - 2 * kin.kappa * e0 <= 0.05:
                    raise ValueError("CT: the radius law cannot be extended before the start: lower the end/start ratio")
                kx = KinShift(kin, e0); dthx = dth + e0 + e1
                nx = int(round(n + e0 / pa0 + e1 / pa1))
                J = int(math.floor(nx - shift - 1e-9))
                d_ = mkdes(kx, dthx, nx, 1e-9)
                ti = lambda K: np.interp(K, d_['kdk'], d_['ktt'])
                if ti(shift) <= e0 + 1e-9 and ti(J) >= e0 + dth - 1e-9:        # every layer has its teeth over the whole usage angle
                    break
            e = e0
            N = max(2, P.layers)
            qs = shift * np.arange(N + 1) / N
            dxs = []; pxs = []
            for i, q in enumerate(qs):
                dq = mkdes(kx, dthx, nx, float(q) if q > 0 else 1e-9)
                dxs.append(dq); pxs.append(make_pair_round(dq, quality, P.clear, rnd))
                wm.progress_update(int(55 + 30 * (i + 1) / (N + 1)))
            ktt, kdk = dxs[0]['ktt'], dxs[0]['kdk']
            tinv = lambda K: np.interp(K, kdk, ktt)
            J = int(math.floor(nx - shift - 1e-9))
            def polar(G):
                ph = np.maximum.accumulate(np.unwrap(np.arctan2(G[:, 1], G[:, 0]))); return ph, np.hypot(G[:, 0], G[:, 1])
            def rack_at(i, v):
                x = np.asarray(kx.s(tinv(v + qs[i]))); R = pxs[i]['rack']; return x, np.interp(x, R[:, 0], R[:, 1])
            def gear_at(i, v):
                ph = tinv(v + qs[i]); g_ph, g_R = polar(pxs[i]['gear']); return ph, np.interp(ph, g_ph, g_R)
            def samples(fn, M):
                """per tooth: points spaced by arc length (step M = the edge length); the same tooth parameters in every layer,
                so the quads run along the tooth"""
                fr = np.linspace(0.0, 1.0, 1601); out = []
                for j in range(J):
                    a_, b_ = fn(0, j + fr)
                    if fn is gear_at:
                        a_ = a_ * b_
                    dxy = np.stack([np.diff(a_), np.diff(b_)], 1); ds = np.hypot(dxy[:, 0], dxy[:, 1])
                    u = np.r_[0.0, np.cumsum(ds)]
                    nm = max(8, int(math.ceil(u[-1] / M)))
                    out.append(j + np.interp(np.arange(nm) / nm * u[-1], u, fr))
                return np.r_[np.concatenate(out), J]
            M = P.min_edge
            vr = samples(rack_at, M)
            vg = samples(gear_at, M)
            racks = [rack_at(i, vr) for i in range(N + 1)]
            gears = [gear_at(i, vg) for i in range(N + 1)]
            # drop the profile vertices where the profile turns by less than the merge angle, checked in every layer
            n_before = (len(vr), len(vg))
            kr = decimate_greedy([np.stack(r_, 1) for r_ in racks], bh / N, P.tol)
            kg = decimate_greedy([np.stack([g_[1] * np.cos(g_[0]), g_[1] * np.sin(g_[0])], 1) for g_ in gears], bh / N, P.tol)
            racks = [(r_[0][kr], r_[1][kr]) for r_ in racks]
            gears = [(g_[0][kg], g_[1][kg]) for g_ in gears]
            # the mesh as it is: interference in a layer and half way between two layers (straight between the layers)
            jm = N // 2
            def mesh_pen(a_, b_):
                rx = 0.5 * (racks[a_][0] + racks[b_][0]); ry = 0.5 * (racks[a_][1] + racks[b_][1])
                ga = [np.stack([gears[c][1] * np.cos(gears[c][0]), gears[c][1] * np.sin(gears[c][0])], 1) for c in (a_, b_)]
                return pair_check(dxs[jm], np.stack([rx, ry], 1), 0.5 * (ga[0] + ga[1]), npos=60, mu_t=P.mu_t, mu_g=P.mu_g)['pen']
            pen_layer = mesh_pen(jm, jm); pen_mid = mesh_pen(jm, min(jm + 1, N))
            # rows: from z = 0 to the middle and back (mirror image = chevron); real coordinates (the turn starts at 0 again)
            order = list(range(N + 1)) + list(range(N - 1, -1, -1))
            zs = [bh * i / N for i in range(N + 1)] + [bh + bh * (j + 1) / N for j in range(N)]
            x_lo = min(float(r_[0][0]) for r_ in racks) - 0.5; x_hi = max(float(r_[0][-1]) for r_ in racks) + 0.5
            yb = min(float(r_[1].min()) for r_ in racks) - P.base
            so = kx.s0
            rack_rows = []
            for i in order:
                x_, y_ = racks[i]
                top = np.vstack([[x_lo, y_[0]], np.stack([x_, y_], 1), [x_hi, y_[-1]]])
                top[:, 0] += so
                rack_rows.append([np.vstack([top, [[x_hi + so, yb], [x_lo + so, yb]]])])
            hub_r = max(0.5 * min(float(g_[1].min()) for g_ in gears), 0.5 * P.bore + 2.0)
            a0 = math.radians(P.start_angle)
            n_arc = max(8, int(math.ceil((2 * math.pi - dthx) / 0.05)))
            hole = None
            if P.bore > 0:
                a = np.linspace(0, 2 * math.pi, 64, endpoint=False)
                hole = np.stack([0.5 * P.bore * np.cos(a), 0.5 * P.bore * np.sin(a)], 1)
            gear_rows = []
            for i in order:
                ph, R_ = gears[i]
                ph = ph - e + a0
                ang = np.linspace(ph[-1], ph[0] + 2 * math.pi, n_arc + 2)[1:-1]
                outer = np.vstack([np.stack([R_ * np.cos(ph), R_ * np.sin(ph)], 1), np.stack([hub_r * np.cos(ang), hub_r * np.sin(ang)], 1)])
                gear_rows.append([outer] + ([hole] if hole is not None else []))
            k = 1
            while bpy.data.objects.get("%s_chev_%d_rack" % (P.case, k)) or bpy.data.objects.get("%s_chev_%d_gear" % (P.case, k)):
                k += 1
            rn, gn = "%s_chev_%d_rack" % (P.case, k), "%s_chev_%d_gear" % (P.case, k)
            ro = bpy.data.objects.new(rn, loft_mesh(rn, rack_rows, zs))
            go = bpy.data.objects.new(gn, loft_mesh(gn, gear_rows, zs))
            context.collection.objects.link(ro); context.collection.objects.link(go)
            kj = dict(kin=({'case': 1, 'F0': kin.F0, 'k': kin.k, 'r0': kin.r0} if P.case == 'CT' else
                           {'case': 2, 'F': kin.F, 'tau0': kin.tau0, 'kth': kin.kth}), dth=dth, a0=a0)
            go['chev'] = repr(kj); ro['chev'] = repr(kj)
            POSE_TARGET[context.scene.name] = (rn, gn, kj)
            apply_pose(context.scene)
        except Exception as e:
            P.report = "ERROR\n%s" % e
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        finally:
            wm.progress_end()
        worst_pen = max(c['pen'] for c in chk); cmin = min(c['contact'] for c in chk)
        L2 = float(eff.min()); loss = float(np.mean([c['loss'] for c in chk]))
        F = pl['F2'] if P.case == 'CT' else pl['F']
        betas = [math.degrees(math.atan(kappa * r)) for r in (pl['rmin'], pl['rmax'])]
        lines = ["%s chevron rack %d | stroke %.1f mm | gear turn %.0f deg | radius %.1f -> %.1f mm | %d teeth" % (
                    P.case, k, pl['S'], P.use_angle, pl['r_start'], pl['r_end'], n),
                 "helix angle %.0f deg at the reference radius %.1f mm (%.0f..%.0f deg over the stroke) | each half shifts %.2f pitch | %d layers per half" % (
                    P.helix, r_ref, min(betas), max(betas), shift, N),
                 "flank angle %s | tooth height %.1f modules | rack tooth %.2f of the pitch | pitch law %s | module %.2f -> %.2f mm" % (
                    ("%.1f deg constant" % awc) if awc is not None else "tilt + %.0f deg" % (delta or 0.0), add_used, thr_used,
                    {0.0: "equal gear angle", 1.0: "equal rack pitch", 2.0: "module ~ force"}[pl['law']], m_lo, m_hi),
                 "printable with a %.2f mm nozzle: every corner r >= %.2f mm | tip width 0.1 mm below the top: rack %.2f, gear %.2f mm (smallest)" % (
                    P.nozzle, 0.5 * P.nozzle, wr.min() if len(wr) else float('nan'), wg.min() if len(wg) else float('nan')),
                 "chevron contact length (2D averaged over the shifted part of a pitch): %.2f mm at the weakest place, %.2f mm mean; dead places %.0f%%" % (
                    eff.min(), eff.mean(), 100 * dead),
                 "mesh interference: in a layer %.0f um, between two layers %.0f um | profile vertices rack %d -> %d, gear %d -> %d (tolerance %.3f mm)" % (
                    1000 * pen_layer, 1000 * pen_mid, n_before[0], len(kr), n_before[1], len(kg), P.tol),
                 "layer check (3 layers, as straight teeth): contact %.0f%% minimum, interference %.1f um" % (100 * cmin, 1000 * worst_pen),
                 "loss in the slice %.1f%% (tooth friction %.2f, guide %.2f); the slope adds sliding along the tooth: about %.1f%% with the helix" % (
                    100 * loss, P.mu_t, P.mu_g, 100 * loss / max(1e-9, math.cos(math.radians(max(betas)))) ),
                 "contact pressure %.1f MPa at the weakest place (largest force / chevron contact length / face width)" % (F / max(L2, 1e-6) / P.width),
                 "axial force of each half %.1f..%.1f N, they cancel in the chevron" % (F * math.tan(math.radians(min(betas))), F * math.tan(math.radians(max(betas))))]
        if P.case == 'CT':
            lines.insert(1, "spring: force %.1f -> %.1f N, k %.4g N/mm, stroke %.1f mm; torque on the gear %.0f N*mm (constant)" % (pl['F1'], pl['F2'], pl['k'], pl['S'], pl['T']))
        else:
            lines.insert(1, "spring: torque %.0f -> %.0f N*mm over %.0f deg, k %.4g N*mm/deg; force on the rack %.1f N (constant)" % (pl['tau1'], pl['tau2'], P.use_angle, pl['k'], pl['F']))
        if 'spring' in pl:
            lines.append("spring helper: " + pl['spring'])
        warn = list(pl['warn']) + notes
        if shift < 1.0:
            warn.append("each half shifts the teeth by %.2f pitch only (1 or more is best): raise the helix angle or the face width" % shift)
        if dead > 0:
            warn.append("the chevron has dead places on %.0f%% of the stroke" % (100 * dead))
        if F / max(L2, 1e-6) / P.width > 15.0:
            warn.append("contact pressure %.0f MPa is high for PLA: widen the face or use a larger module" % (F / max(L2, 1e-6) / P.width))
        if pen_mid > 0.05:
            warn.append("between the layers the teeth interfere by %.0f um: more layers per half" % (1000 * pen_mid))
        if worst_pen >= (0.01 if quality == 'fine' else 0.03):
            warn.append("interference %.0f um in a slice" % (1000 * worst_pen))
        P.report = "\n".join(lines + ["WARNING: " + w for w in warn] + ["built in %.0f s" % (time.time() - t0)])
        self.report({'INFO'}, "chevron rack pair created")
        return {'FINISHED'}

class CHEV_PT_panel(bpy.types.Panel):
    bl_label = "Chevron rack"
    bl_idname = "CHEV_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Chevron rack"

    def draw(self, context):
        lay = self.layout
        P = context.scene.chev
        lay.prop(P, "case", expand=True)
        b = lay.box(); b.label(text="Working range", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        b.prop(P, "r_start"); b.prop(P, "f_start"); b.prop(P, "ratio"); b.prop(P, "use_angle"); b.prop(P, "start_angle")
        pl = plan_ends(P)
        res = lay.box()
        if pl['ok']:
            if P.case == 'CT':
                res.label(text="force %.1f -> %.1f N | k %.4g N/mm | stroke %.1f mm" % (pl['F1'], pl['F2'], pl['k'], pl['S']))
                res.label(text="torque on the gear %.0f N*mm | radius %.1f -> %.1f mm" % (pl['T'], pl['r_start'], pl['r_end']))
            else:
                res.label(text="torque %.0f -> %.0f N*mm | k %.4g N*mm/deg | wound %.0f deg" % (pl['tau1'], pl['tau2'], pl['k'], P.use_angle))
                res.label(text="force on the rack %.1f N | stroke %.1f mm | radius %.1f -> %.1f mm" % (pl['F'], pl['S'], pl['r_start'], pl['r_end']))
            res.label(text="%d teeth | smallest module %.2f mm | tilt of the pitch curve up to %.0f deg" % (pl['n'], pl['m_min'], pl['tilt_max']))
            if 'spring' in pl:
                res.label(text=pl['spring'])
        for e in pl['err']:
            res.label(text=e, icon='ERROR')
        for w in pl['warn']:
            res.label(text=w, icon='INFO')
        s = lay.box(); s.label(text="Spring helper (optional)", icon='MOD_SCREW')
        s.prop(P, "sp_d"); s.prop(P, "sp_D"); s.prop(P, "sp_mod"); s.prop(P, "sp_allow")
        t = lay.box(); t.label(text="Teeth", icon='MOD_BUILD')
        t.prop(P, "m_min"); t.prop(P, "add"); t.prop(P, "law"); t.prop(P, "thr_mode")
        if P.thr_mode == 'FIXED':
            t.prop(P, "thr")
        t.prop(P, "nozzle"); t.prop(P, "angle_mode")
        if P.angle_mode == 'OFFSET':
            t.prop(P, "offset")
        elif P.angle_mode == 'CONST':
            t.prop(P, "alpha")
        t.prop(P, "back")
        c = lay.box(); c.label(text="Chevron teeth", icon='MOD_ARRAY')
        c.prop(P, "helix"); c.prop(P, "helix_at"); c.prop(P, "layers"); c.prop(P, "min_edge"); c.prop(P, "tol"); c.prop(P, "width")
        o = lay.box()
        o.prop(P, "base"); o.prop(P, "bore"); o.prop(P, "clear")
        row = o.row(); row.prop(P, "mu_t"); row.prop(P, "mu_g")
        o.prop(P, "quality")
        lay.operator("chev.generate", icon='MOD_BUILD')
        lay.prop(P, "pose")
        if P.report:
            rb = lay.box()
            for line in P.report.split("\n"):
                rb.label(text=line, icon=('ERROR' if line.startswith('ERROR') or line.startswith('WARNING') else 'NONE'))

classes = (ChevProps, CHEV_OT_generate, CHEV_PT_panel)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.chev = PointerProperty(type=ChevProps)

def unregister():
    del bpy.types.Scene.chev
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
