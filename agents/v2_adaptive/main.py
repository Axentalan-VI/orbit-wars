"""v2 adaptive bot for Orbit Wars.

Key improvements over v1:
  1. Target-centric planning  (best source per target, avoids duplicate attacks)
  2. Own-fleet tracking       (skips targets already covered by in-flight ships)
  3. ROI-based scoring        (production * remaining_turns / cost)
  4. Minimal force dispatch   (req + 1 instead of req + SAFETY)
  5. Game-phase awareness     (rush neutrals early, consolidate late)
  6. Comet priority boost
  7. Smart defense            (reserve only for real threats; evacuate doomed planets)
  8. Reinforcement            (redistribute ships to threatened allies)
  9. Iterative ETA refinement (accounts for smaller fleet being slower)
 10. Aggression calibration   (push harder when behind, hold when ahead)
"""

import math

# ---------- constants ----------
BOARD = 100.0
SUN_R = 10.0
MAX_SPD = 6.0
ROT_LIM = 50.0

# ---------- persistent game state ----------
_g = {}


def _spd(n):
    if n <= 1:
        return 1.0
    t = min(1.0, max(0.0, math.log(n) / math.log(1000)))
    return 1.0 + (MAX_SPD - 1.0) * t ** 1.5


def _orb(x, y, r):
    return math.hypot(x - 50, y - 50) + r < ROT_LIM


def _pos(x0, y0, r, av, s):
    if not _orb(x0, y0, r):
        return x0, y0
    d = math.hypot(x0 - 50, y0 - 50)
    th = math.atan2(y0 - 50, x0 - 50) + av * s
    return 50 + d * math.cos(th), 50 + d * math.sin(th)


def _sun(sx, sy, tx, ty):
    r = SUN_R + 1.0
    dx, dy = tx - sx, ty - sy
    fx, fy = sx - 50, sy - 50
    a = dx * dx + dy * dy
    if a == 0:
        return fx * fx + fy * fy <= r * r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _arr(sx, sy, n, ix, iy, ir, av, step):
    """ETA + predicted arrival (x,y) for *n* ships launched from (sx,sy)."""
    sp = _spd(n)
    tx, ty = _pos(ix, iy, ir, av, step)
    eta = 1
    for _ in range(6):
        d = math.hypot(sx - tx, sy - ty)
        eta = max(1, int(math.ceil(d / sp)))
        nx, ny = _pos(ix, iy, ir, av, step + eta)
        if abs(nx - tx) + abs(ny - ty) < 0.01:
            break
        tx, ty = nx, ny
    return eta, tx, ty


def _init(obs):
    ini = obs.get("initial_planets") or []
    key = tuple((p[0],) for p in ini[:5])
    if _g.get("key") != key:
        _g.clear()
        _g["key"] = key
        _g["ini"] = {int(p[0]): p for p in ini}


# ------------------------------------------------------------------
def agent(obs, config=None):
    _init(obs)
    P = obs.get("planets") or []
    F = obs.get("fleets") or []
    me = int(obs.get("player", 0))
    av = float(obs.get("angular_velocity", 0))
    step = int(obs.get("step", 0))
    cids = set(int(c) for c in (obs.get("comet_planet_ids") or []))
    INI = _g.get("ini", {})

    my = [p for p in P if int(p[1]) == me]
    if not my:
        return []

    turns_left = max(1, 500 - step)
    early = step < 80
    late = step > 380

    # ===== 1. Own fleet target tracking =====
    own_to = {}  # target_pid -> own ships already in-flight
    for f in F:
        if int(f[1]) != me:
            continue
        fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), int(f[6])
        sp = _spd(fs)
        vx, vy = math.cos(fa) * sp, math.sin(fa) * sp
        bp, bt = None, 70
        for p in P:
            pid, pr = int(p[0]), float(p[4])
            ip = INI.get(pid)
            cx, cy = (_pos(ip[2], ip[3], ip[4], av, step + 1)
                       if ip else (float(p[2]), float(p[3])))
            if vx * (cx - fx) + vy * (cy - fy) <= 0:
                continue
            for t in range(1, min(bt, 55)):
                nx, ny = fx + vx * t, fy + vy * t
                if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                    break
                cx, cy = (_pos(ip[2], ip[3], ip[4], av, step + t)
                           if ip else (float(p[2]), float(p[3])))
                if math.hypot(nx - cx, ny - cy) <= pr + 0.5:
                    bt = t
                    bp = pid
                    break
        if bp is not None:
            own_to[bp] = own_to.get(bp, 0) + fs

    # ===== 2. Incoming threat detection =====
    threat = {}
    threat_eta = {}
    for f in F:
        if int(f[1]) == me:
            continue
        fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), int(f[6])
        sp = _spd(fs)
        vx, vy = math.cos(fa) * sp, math.sin(fa) * sp
        for p in my:
            pid, pr = int(p[0]), float(p[4])
            ip = INI.get(pid)
            if not ip:
                continue
            for t in range(1, 30):
                nx, ny = fx + vx * t, fy + vy * t
                if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                    break
                cx, cy = _pos(ip[2], ip[3], ip[4], av, step + t)
                if math.hypot(nx - cx, ny - cy) <= pr + 1.0:
                    threat[pid] = threat.get(pid, 0) + fs
                    prev = threat_eta.get(pid)
                    if prev is None or t < prev:
                        threat_eta[pid] = t
                    break

    # ===== 3. Reserve, surplus, doomed-planet detection =====
    doomed = set()
    reserve = {}
    for p in my:
        pid = int(p[0])
        ships, prod = int(p[5]), int(p[6])
        t_ships = threat.get(pid, 0)
        t_eta = threat_eta.get(pid, 99)
        if t_ships > 0:
            can_hold = ships + prod * t_eta
            if t_ships > can_hold * 1.4:
                doomed.add(pid)
                reserve[pid] = 0          # evacuate!
            else:
                reserve[pid] = min(ships, t_ships + 2)
        else:
            reserve[pid] = 1              # minimal garrison
    surplus = {int(p[0]): max(0, int(p[5]) - reserve[int(p[0])]) for p in my}

    # ===== 4. Weighted centre of mass =====
    tp = sum(int(p[6]) for p in my) or 1
    cmx = sum(float(p[2]) * int(p[6]) for p in my) / tp
    cmy = sum(float(p[3]) * int(p[6]) for p in my) / tp

    # ===== 5. Strength estimate =====
    my_ships = (sum(int(p[5]) for p in my)
                + sum(int(f[6]) for f in F if int(f[1]) == me))
    en_ships = (sum(int(p[5]) for p in P if int(p[1]) not in (me, -1))
                + sum(int(f[6]) for f in F if int(f[1]) not in (me, -1)))

    # ===== 6. Target-centric candidate evaluation =====
    cands = []
    for tgt in P:
        tid, tow = int(tgt[0]), int(tgt[1])
        if tow == me:
            continue
        tprod = max(1, int(tgt[6]))
        tships = int(tgt[5])
        is_comet = tid in cids
        enr = own_to.get(tid, 0)

        # Skip if already sending enough
        if tow == -1 and enr > tships + 3:
            continue
        if tow >= 0 and tow != me and enr > tships + tprod * 10 + 3:
            continue

        ip = INI.get(tid)
        if not ip:
            ip = [tid, -1, float(tgt[2]), float(tgt[3]),
                  float(tgt[4]), 0, tprod]

        for src in my:
            sid = int(src[0])
            if surplus[sid] < 2:
                continue
            sx, sy = float(src[2]), float(src[3])

            # ---- first ETA pass (speed upper-bound = surplus ships) ----
            eta1, ax1, ay1 = _arr(sx, sy, surplus[sid],
                                  ip[2], ip[3], ip[4], av, step)
            if eta1 <= 0 or eta1 > 180:
                continue
            if late and eta1 > 25:
                continue
            if _sun(sx, sy, ax1, ay1):
                continue

            # required ships for this ETA
            if tow == -1:
                req = tships + 1 - enr
            else:
                req = tships + tprod * eta1 + 1 - enr
            req = max(1, req)
            ships = req + 1           # +1 minimal safety
            if ships > surplus[sid]:
                continue

            # ---- second ETA pass (actual fleet size → accurate speed) ----
            eta2, ax2, ay2 = _arr(sx, sy, ships,
                                  ip[2], ip[3], ip[4], av, step)
            if _sun(sx, sy, ax2, ay2):
                continue
            # recompute required with refined ETA
            if tow == -1:
                req2 = tships + 1 - enr
            else:
                req2 = tships + tprod * eta2 + 1 - enr
            req2 = max(1, req2)
            ships2 = req2 + 1
            if ships2 > surplus[sid]:
                continue

            # ---- ROI scoring ----
            prod_gain = tprod * max(0, turns_left - eta2)
            roi = prod_gain / ships2

            # neutral bonus (cheaper: no production during transit)
            if tow == -1:
                roi *= 2.0 if early else 1.3

            # comet bonus
            if is_comet:
                roi *= 1.4

            # proximity bonus (keep territory compact)
            roi -= math.hypot(ax2 - cmx, ay2 - cmy) * 0.025

            # contest penalty (enemy fleets heading toward same target)
            for ef in F:
                if int(ef[1]) == me:
                    continue
                edx, edy = math.cos(ef[4]), math.sin(ef[4])
                to = (ax2 - ef[2], ay2 - ef[3])
                n = math.hypot(*to)
                if n > 0 and (edx * to[0] + edy * to[1]) / n > 0.7:
                    roi -= int(ef[6]) * 0.12

            # aggression boost when behind
            if en_ships > my_ships * 1.3 and tow >= 0 and tow != me:
                roi *= 1.3

            # evacuation: doomed planet ships should attack instead of dying
            if sid in doomed:
                roi += 8.0

            angle = math.atan2(ay2 - sy, ax2 - sx)
            cands.append((roi, sid, tid, ships2, angle))

    cands.sort(key=lambda c: c[0], reverse=True)

    # ===== 7. Greedy assignment (one fleet per target) =====
    actions = []
    used_t = set()
    for roi, sid, tid, ships, angle in cands:
        if roi <= 0:
            break
        if tid in used_t:
            continue
        if surplus[sid] < ships:
            continue
        actions.append([sid, float(angle), int(ships)])
        surplus[sid] -= ships
        used_t.add(tid)
        if sum(surplus.values()) < 2:
            break

    # ===== 8. Reinforcement: safe → threatened allies =====
    if not late:
        for src in my:
            sid = int(src[0])
            if surplus[sid] < 8:
                continue
            if threat.get(sid, 0) > 0:
                continue
            bt, bu = None, 0
            for tgt in my:
                td = int(tgt[0])
                if td == sid:
                    continue
                u = threat.get(td, 0) - int(tgt[5])
                if u > bu:
                    bu = u
                    bt = tgt
            if bt and bu > 3:
                # aim at predicted future position of reinforcement target
                ip_r = INI.get(int(bt[0]))
                sx, sy = float(src[2]), float(src[3])
                send = min(surplus[sid], bu + 3)
                if send >= 4:
                    if ip_r:
                        eta_r, rx, ry = _arr(sx, sy, send,
                                             ip_r[2], ip_r[3], ip_r[4],
                                             av, step)
                    else:
                        rx, ry = float(bt[2]), float(bt[3])
                    if not _sun(sx, sy, rx, ry):
                        a = math.atan2(ry - sy, rx - sx)
                        actions.append([sid, float(a), int(send)])
                        surplus[sid] -= send

    return actions
