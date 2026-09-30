"""v16_hoard for Orbit Wars.

Restraint-economy + late-burst strategy.

Phases:
  OPENING (step <  80): Capture only cheap nearby NEUTRALS (incl. comets).
  HOARD   (step < 380): No offensive fleets. Defend hard. Comets only.
  BURST   (step >=380): Identify weakest opponent by total ship count and
                        snipe their planets with calculated overkill.

The bet: while v0-v7 style bots slug it out from turn 1, depleting each
other, v16 sits quietly accumulating production. By the time we engage,
opponents are damaged. Even if we capture nothing, hoarding mean we
finish with high ship totals and good placement.

Defense pass runs every turn in every phase (using the same ledger
logic as our older agents).

Action format: list of [source_planet_id, angle, ships].
"""

import math

BOARD = 100.0
SUN_R = 10.0
MAX_SPD = 6.0
ROT_LIM = 50.0
COMET_SPAWNS = (50, 150, 250, 350, 450)

# Phase boundaries
OPENING_END = 80
BURST_START = 380

# Defense look-ahead
H_DEF = 40

# Per-phase ETA caps (longer in BURST so far planets can still strike)
H_ATK_OPENING = 50
H_ATK_BURST = 90

# Speed-overkill: in race phases we send max(req+1, surplus * RACE_FRAC)
# so the fleet is large => moves faster (per the engine's log-speed curve).
# A 5-ship fleet moves at ~2.0/turn; a 50-ship fleet moves at ~4.4/turn.
RACE_FRAC = 0.6

_g = {}


# ---------------------------------------------------------------------------
# Physics helpers (engine-mirrored, NOT strategy)
# ---------------------------------------------------------------------------
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


def _sun_hit(sx, sy, tx, ty, r=SUN_R + 1.0):
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
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _pred_pos(pid, ini, comet_tracks, av, step):
    ct = comet_tracks.get(pid)
    if ct is not None:
        idx_now, path = ct
        # path is indexed from path_index=0 at game start? No: ct stores
        # (current path_index, full path). Caller passes absolute step; we
        # need offset from path's "current" index. But the engine gives
        # path_index = current step within path. Treating step here as
        # absolute step requires knowing path origin. We keep the same
        # convention as v7: indices into path are absolute steps in path.
        # Path entries align with game step under the engine spec, and
        # path_index advances with the game. Use direct step index.
        if 0 <= step < len(path):
            p = path[step]
            return float(p[0]), float(p[1])
        return None
    ip = ini.get(pid)
    if ip is None:
        return None
    return _pos(ip[2], ip[3], ip[4], av, step)


def _arr_eta(sx, sy, ships, get_pos, current_step):
    """Fixed-point ETA estimator for a moving target."""
    sp = _spd(ships)
    init = get_pos(current_step + 1)
    if init is None:
        return None, None, None
    tx, ty = init
    eta = 1
    for _ in range(6):
        d = math.hypot(sx - tx, sy - ty)
        neta = max(1, int(math.ceil(d / sp)))
        npos = get_pos(current_step + neta)
        if npos is None:
            return None, None, None
        nx, ny = npos
        if neta == eta and abs(nx - tx) + abs(ny - ty) < 0.02:
            tx, ty, eta = nx, ny, neta
            break
        tx, ty, eta = nx, ny, neta
    return eta, tx, ty


def _best_angle(sx, sy, ax, ay):
    base = math.atan2(ay - sy, ax - sx)
    if not _sun_hit(sx, sy, ax, ay):
        return base, ax, ay
    d = math.hypot(ax - sx, ay - sy)
    for delta in (0.35, -0.35, 0.7, -0.7, 1.05, -1.05):
        a = base + delta
        tx = sx + d * math.cos(a)
        ty = sy + d * math.sin(a)
        if not _sun_hit(sx, sy, tx, ty):
            return a, tx, ty
    return None, None, None


def _predict_fleet_target(f, P, ini, comet_tracks, av, step, max_t=80):
    fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), int(f[6])
    sp = _spd(fs)
    vx, vy = math.cos(fa) * sp, math.sin(fa) * sp
    sun_blocked_t = None
    for t in range(1, max_t):
        nx, ny = fx + vx * t, fy + vy * t
        if math.hypot(nx - 50, ny - 50) <= SUN_R:
            sun_blocked_t = t
            break
        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            break
    best_pid, best_t = None, max_t
    for p in P:
        pid, pr = int(p[0]), float(p[4])
        cur_pos = _pred_pos(pid, ini, comet_tracks, av, step + 1)
        if cur_pos is None:
            continue
        if vx * (cur_pos[0] - fx) + vy * (cur_pos[1] - fy) <= 0:
            continue
        for t in range(1, best_t):
            if sun_blocked_t is not None and t >= sun_blocked_t:
                break
            nx, ny = fx + vx * t, fy + vy * t
            if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                break
            tpos = _pred_pos(pid, ini, comet_tracks, av, step + t)
            if tpos is None:
                break
            if math.hypot(nx - tpos[0], ny - tpos[1]) <= pr + 0.5:
                best_t = t
                best_pid = pid
                break
    return best_pid, (best_t if best_pid is not None else max_t)


# ---------------------------------------------------------------------------
# Game-state accessors
# ---------------------------------------------------------------------------
def _init_game(obs):
    ini = obs.get("initial_planets") or []
    key = tuple(
        (int(p[0]), round(float(p[2]), 2), round(float(p[3]), 2))
        for p in ini[:6]
    )
    if _g.get("key") != key:
        _g.clear()
        _g["key"] = key
        _g["ini"] = {int(p[0]): p for p in ini}


def _build_comet_tracks(obs):
    tracks = {}
    for grp in obs.get("comets") or []:
        ids = grp.get("planet_ids") or []
        paths = grp.get("paths") or []
        idx = int(grp.get("path_index", 0))
        for pid, path in zip(ids, paths):
            tracks[int(pid)] = (idx, path)
    return tracks


def _build_arrivals(F, P, ini, comet_tracks, av, step):
    arrivals = {int(p[0]): [] for p in P}
    for f in F:
        pid, t = _predict_fleet_target(f, P, ini, comet_tracks, av, step)
        if pid is None:
            continue
        arrivals[pid].append((t, int(f[1]), int(f[6])))
    for pid in arrivals:
        arrivals[pid].sort()
    return arrivals


def _defense_reserve(p, arrivals, me):
    """Compute (reserve, max_threat, doomed) for an owned planet."""
    pid = int(p[0])
    g0, prod = int(p[5]), int(p[6])
    arr = arrivals.get(pid, [])
    garrison = g0
    owner = me
    en_cum = my_cum = 0
    earliest_loss = None
    prev_t = 0
    max_def = 0
    for (t, ow, s) in arr:
        if t > H_DEF:
            break
        garrison += prod * (t - prev_t)
        if ow == me:
            my_cum += s
            if owner == me:
                garrison += s
            else:
                if s > garrison:
                    owner = me
                    garrison = s - garrison
                else:
                    garrison -= s
        else:
            en_cum += s
            if owner == me:
                if s > garrison:
                    owner = ow
                    garrison = s - garrison
                    if earliest_loss is None:
                        earliest_loss = t
                else:
                    garrison -= s
        prev_t = t
        max_def = max(max_def, en_cum - my_cum - g0 - prod * t)
    doomed = (
        earliest_loss is not None
        and en_cum > (g0 + prod * earliest_loss + my_cum) * 1.3
    )
    if doomed:
        return 0, max_def, True
    return min(g0, max(1, max_def + 2)), max_def, False


def _required_for(tgt, eta, arrivals, me):
    tid, tow = int(tgt[0]), int(tgt[1])
    tprod = max(1, int(tgt[6]))
    garrison = int(tgt[5])
    owner = tow
    arr = arrivals.get(tid, [])
    prev_t = 0
    for (t, ow, s) in arr:
        if t >= eta:
            break
        if owner >= 0:
            garrison += tprod * (t - prev_t)
        if ow == owner:
            garrison += s
        else:
            if s > garrison:
                owner = ow
                garrison = s - garrison
            else:
                garrison -= s
        prev_t = t
    if owner >= 0:
        garrison += tprod * (eta - prev_t)
    if owner == me:
        return 0
    return garrison + 1


def _plan_attack(src, tgt, surplus, arrivals, ini, comet_tracks, av, step,
                 me, max_eta, race=False):
    sid = int(src[0])
    if surplus[sid] < 2:
        return None
    sx, sy = float(src[2]), float(src[3])

    def get_pos(s):
        return _pred_pos(int(tgt[0]), ini, comet_tracks, av, s)

    eta1, ax, ay = _arr_eta(sx, sy, surplus[sid], get_pos, step)
    if eta1 is None or eta1 > max_eta:
        return None
    angle, ax, ay = _best_angle(sx, sy, ax, ay)
    if angle is None:
        return None
    req = _required_for(tgt, eta1, arrivals, me)
    if req == 0:
        return None
    # In race phases, send extra to move faster (log-scaled speed curve).
    if race:
        ships = max(req + 1, int(surplus[sid] * RACE_FRAC))
    else:
        ships = req + 1
    if ships > surplus[sid]:
        ships = surplus[sid]
    if ships < req + 1:
        return None
    eta2, ax, ay = _arr_eta(sx, sy, ships, get_pos, step)
    if eta2 is None or eta2 > max_eta:
        return None
    angle2, ax, ay = _best_angle(sx, sy, ax, ay)
    if angle2 is None:
        return None
    req2 = _required_for(tgt, eta2, arrivals, me)
    if req2 == 0:
        return None
    # Recompute ships for the (possibly updated) eta2.
    if race:
        ships2 = max(req2 + 1, int(surplus[sid] * RACE_FRAC))
    else:
        ships2 = req2 + 1
    if ships2 > surplus[sid]:
        ships2 = surplus[sid]
    if ships2 < req2 + 1:
        return None
    # Comet path-end safety
    ct = comet_tracks.get(int(tgt[0]))
    if ct is not None:
        remaining = len(ct[1]) - ct[0] - 1
        if eta2 > remaining - 3:
            return None
    return ships2, angle2, eta2


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def agent(obs, config=None):
    _init_game(obs)
    P = obs.get("planets") or []
    F = obs.get("fleets") or []
    me = int(obs.get("player", 0))
    av = float(obs.get("angular_velocity", 0))
    step = int(obs.get("step", 0))
    ini = _g.get("ini", {})
    comet_tracks = _build_comet_tracks(obs)
    cids = set(int(c) for c in (obs.get("comet_planet_ids") or []))

    my = [p for p in P if int(p[1]) == me]
    if not my:
        return []

    arrivals = _build_arrivals(F, P, ini, comet_tracks, av, step)

    # ---------- Defense pass: compute surplus per planet ----------
    surplus = {}
    doomed = set()
    for p in my:
        pid = int(p[0])
        reserve, _max_def, dm = _defense_reserve(p, arrivals, me)
        if dm:
            doomed.add(pid)
        surplus[pid] = max(0, int(p[5]) - reserve)

    # ---------- Phase ----------
    if step < OPENING_END:
        phase = "OPENING"
    elif step < BURST_START:
        phase = "HOARD"
    else:
        phase = "BURST"

    # Opponent strength (used in BURST)
    weakest = None
    if phase == "BURST":
        opp_power = {}
        for p in P:
            ow = int(p[1])
            if ow >= 0 and ow != me:
                opp_power[ow] = opp_power.get(ow, 0) + int(p[5])
        for f in F:
            ow = int(f[1])
            if ow >= 0 and ow != me:
                opp_power[ow] = opp_power.get(ow, 0) + int(f[6])
        if opp_power:
            weakest = min(opp_power, key=opp_power.get)

    # ---------- Offensive candidates ----------
    candidates = []  # (priority, sid, tid, ships, angle, eta, tow)

    for tgt in P:
        tid = int(tgt[0])
        tow = int(tgt[1])
        if tow == me:
            continue
        is_comet = tid in cids

        # Phase-specific filtering
        if phase == "OPENING":
            if tow != -1:
                continue  # neutrals only
        elif phase == "HOARD":
            if not is_comet:
                continue  # comets only
        # BURST: anything not ours

        max_eta = (
            H_ATK_OPENING if phase == "OPENING"
            else H_ATK_BURST if phase == "BURST"
            else 80
        )

        for src in my:
            sid = int(src[0])
            if sid in doomed:
                continue
            plan = _plan_attack(
                src, tgt, surplus, arrivals,
                ini, comet_tracks, av, step, me, max_eta,
                race=(phase != "HOARD"),
            )
            if plan is None:
                continue
            ships, angle, eta = plan
            tprod = max(1, int(tgt[6]))
            turns_left = max(1, 500 - step)
            roi = tprod * max(0, turns_left - eta) / max(1, ships)

            # Comets are precious (pay big over the path)
            if is_comet:
                roi *= 1.8

            # In BURST, strongly prefer the weakest opponent
            if phase == "BURST" and tow >= 0:
                if tow == weakest:
                    roi *= 2.0
                else:
                    roi *= 0.4
            # In BURST, neutrals are fine (free production)
            if phase == "BURST" and tow == -1:
                roi *= 1.2

            candidates.append([roi, sid, tid, ships, angle, eta, tow])

    candidates.sort(key=lambda c: c[0], reverse=True)

    actions = []
    target_committed = set()  # one wave per target per turn
    for roi, sid, tid, ships, angle, eta, _tow in candidates:
        if roi <= 0:
            break
        if surplus[sid] < ships:
            continue
        if tid in target_committed:
            continue
        actions.append([sid, float(angle), int(ships)])
        surplus[sid] -= ships
        target_committed.add(tid)
        if sum(surplus.values()) < 2:
            break

    # ---------- Defensive reinforcement ----------
    if step < 480:
        for tgt in my:
            tid = int(tgt[0])
            if tid in doomed:
                continue
            arr = arrivals.get(tid, [])
            if not arr:
                continue
            enemy_total = sum(s for (_t, o, s) in arr if o != me and _t <= H_DEF)
            my_inbound = sum(s for (_t, o, s) in arr if o == me and _t <= H_DEF)
            garrison = int(tgt[5]) + my_inbound
            prod = int(tgt[6])
            if enemy_total <= garrison + prod * H_DEF:
                continue
            need = enemy_total - garrison - prod * 5
            if need <= 2:
                continue
            earliest_en = min(
                (t for (t, o, _s) in arr if o != me),
                default=H_DEF,
            )
            donors = []
            for src in my:
                sid = int(src[0])
                if sid == tid or surplus[sid] < 4:
                    continue
                src_arr = arrivals.get(sid, [])
                if any(o != me for (_t, o, _s) in src_arr[:3]):
                    continue
                sx, sy = float(src[2]), float(src[3])

                def get_pos(s, _tid=tid):
                    return _pred_pos(_tid, ini, comet_tracks, av, s)

                eta_r, rx, ry = _arr_eta(sx, sy, surplus[sid], get_pos, step)
                if eta_r is None or eta_r >= earliest_en:
                    continue
                ang, _, _ = _best_angle(sx, sy, rx, ry)
                if ang is None:
                    continue
                donors.append((eta_r, sid, ang))
            donors.sort()
            remaining = need
            for eta_r, sid, ang in donors:
                if remaining <= 0:
                    break
                send = min(surplus[sid] - 1, remaining + 2)
                if send < 3:
                    continue
                actions.append([sid, float(ang), int(send)])
                surplus[sid] -= send
                remaining -= send

    return actions
