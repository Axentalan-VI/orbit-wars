"""v1 heuristic bot for Orbit Wars.

Rule-based. Self-contained (stdlib only) so the file can be submitted directly
or bundled by ``scripts/package_agent.py`` without extra assets.

Strategy per turn:
  1. Predict positions/threats for next ~20 turns.
  2. For each owned planet, compute ``need`` = incoming enemy ships - local
     garrison over the next few turns. Reserve that many defensive ships.
  3. Score every (source, target) pair by a weighted combo of production,
     arrival time, ships-to-take, and contested penalty. Dispatch greedy
     best-first using surplus (garrison - reserved) ships.
"""

import math

# ---------------------------------------------------------------------------
# Inlined rules helpers (kept in sync with rules.py). Submissions must be
# self-contained - no local package imports.
# ---------------------------------------------------------------------------
BOARD = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
MAX_SHIP_SPEED = 6.0
ROTATION_RADIUS_LIMIT = 50.0


def _fleet_speed(ships: float, max_speed: float = MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    t = math.log(ships) / math.log(1000.0)
    t = max(0.0, min(1.0, t))
    return 1.0 + (max_speed - 1.0) * (t ** 1.5)


def _is_orbiting(x: float, y: float, r: float) -> bool:
    return math.hypot(x - CENTER[0], y - CENTER[1]) + r < ROTATION_RADIUS_LIMIT


def _predict_pos(x0, y0, r, av, step):
    if not _is_orbiting(x0, y0, r):
        return x0, y0
    cx, cy = CENTER
    dx, dy = x0 - cx, y0 - cy
    rad = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) + av * step
    return cx + rad * math.cos(theta), cy + rad * math.sin(theta)


def _segment_hits(p0, p1, c, r):
    (x0, y0), (x1, y1), (cx, cy) = p0, p1, c
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - cx, y0 - cy
    a = dx * dx + dy * dy
    if a == 0.0:
        return (fx * fx + fy * fy) <= r * r
    b = 2.0 * (fx * dx + fy * dy)
    c2 = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c2
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 < t2)


def _eta(from_xy, ships, tx0, ty0, r, av, step, max_iter=4):
    """Estimate turns to reach a (possibly orbiting) target at initial (tx0,ty0)."""
    speed = _fleet_speed(ships)
    tx, ty = _predict_pos(tx0, ty0, r, av, step)
    eta = 1
    for _ in range(max_iter):
        d = math.hypot(from_xy[0] - tx, from_xy[1] - ty)
        eta = max(1, int(math.ceil(d / speed)))
        nx, ny = _predict_pos(tx0, ty0, r, av, step + eta)
        if abs(nx - tx) + abs(ny - ty) < 1e-3:
            tx, ty = nx, ny
            break
        tx, ty = nx, ny
    return eta, tx, ty


# ---------------------------------------------------------------------------
# Agent state (per-game cache keyed by initial planets reference)
# ---------------------------------------------------------------------------
_state = {"initial_key": None, "initial_by_id": {}}


def _ensure_state(obs):
    initial = obs.get("initial_planets") or []
    key = len(initial), tuple((p[0], p[2], p[3]) for p in initial[:4])
    if _state["initial_key"] != key:
        _state["initial_key"] = key
        _state["initial_by_id"] = {int(p[0]): p for p in initial}


# ---------------------------------------------------------------------------
# Scoring weights (tuned by hand; easy knobs for later self-play)
# ---------------------------------------------------------------------------
W_PROD = 40.0       # per production point
W_ETA = 1.0         # per turn of travel
W_COST = 0.25       # per ship we have to send
W_CONTEST = 5.0     # per enemy ship expected to intercept before us
W_NEUTRAL = 15.0    # bonus for grabbing a neutral (vs enemy)
W_HOMEWARD = 2.0    # bonus for staying near center-of-mass
SAFETY = 2          # extra ships above the required capture count


def _incoming_threats(planets, fleets, me, av, step_now, horizon=30):
    """For each of my planets, sum enemy ship counts that will arrive within
    ``horizon`` turns. Also returns a per-planet earliest arrival step.

    Uses a simple proximity test: fleet reaches planet if its path comes within
    the planet's radius at some future step (continuous-collision approximation
    is skipped for speed; use linear distance to estimated future planet pos).
    """
    threat = {}
    eta_map = {}
    for f in fleets:
        fid, fowner, fx, fy, fangle, _src, fships = f
        if fowner == me:
            continue
        speed = _fleet_speed(fships)
        vx, vy = math.cos(fangle) * speed, math.sin(fangle) * speed
        for p in planets:
            pid, powner, px0, py0, prad, _pships, _prod = p
            if powner != me:
                continue
            init = _state["initial_by_id"].get(int(pid))
            if init is None:
                continue
            # Try turns 1..horizon
            for t in range(1, horizon + 1):
                nx = fx + vx * t
                ny = fy + vy * t
                if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                    break
                tx, ty = _predict_pos(init[2], init[3], init[4], av, step_now + t)
                if math.hypot(nx - tx, ny - ty) <= prad + 1.0:
                    threat[int(pid)] = threat.get(int(pid), 0) + int(fships)
                    prev = eta_map.get(int(pid))
                    if prev is None or t < prev:
                        eta_map[int(pid)] = t
                    break
    return threat, eta_map


def _required_to_take(planet, eta, me) -> int:
    """How many ships an attacker needs to own the planet upon arrival."""
    _pid, powner, _x, _y, _r, ships, prod = planet
    garrison = int(ships)
    if powner == me:
        return 0
    if powner == -1:
        # Neutrals do not produce -> static garrison.
        return garrison + 1
    # Enemy produces while we're in transit.
    return garrison + int(prod) * int(eta) + 1


def agent(obs, config=None):
    _ensure_state(obs)
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    me = int(obs.get("player", 0))
    av = float(obs.get("angular_velocity", 0.0))
    step_now = int(obs.get("step", 0)) if isinstance(obs, dict) else 0

    mine = [p for p in planets if int(p[1]) == me]
    if not mine:
        return []

    threats, _threat_eta = _incoming_threats(planets, fleets, me, av, step_now)

    # Reserve defensive ships per owned planet.
    reserve = {int(p[0]): min(int(p[5]), threats.get(int(p[0]), 0) + SAFETY)
               for p in mine}
    surplus = {int(p[0]): max(0, int(p[5]) - reserve[int(p[0])]) for p in mine}

    # Center of mass of my planets (homeward bias)
    if mine:
        cmx = sum(p[2] for p in mine) / len(mine)
        cmy = sum(p[3] for p in mine) / len(mine)
    else:
        cmx = cmy = BOARD / 2

    # Candidate moves: (score, src_pid, tgt_pid, ships, angle)
    candidates = []
    for src in mine:
        sid = int(src[0])
        if surplus[sid] < 2:
            continue
        sx, sy, srad = src[2], src[3], src[4]
        for tgt in planets:
            tid, towner, tx0, ty0, trad, _ts, tprod = tgt
            if int(tid) == sid:
                continue
            init = _state["initial_by_id"].get(int(tid))
            if init is None:
                # Comet or dynamic object without initial record - use current pos.
                init = [tid, -1, tx0, ty0, trad, 0, max(1, int(tprod))]
            eta, arr_x, arr_y = _eta((sx, sy), max(2, surplus[sid]),
                                       init[2], init[3], init[4], av, step_now)
            if eta <= 0 or eta > 250:
                continue
            # Reject if path crosses the sun (approximate: check line to arrival pos).
            if _segment_hits((sx, sy), (arr_x, arr_y), CENTER, SUN_RADIUS + 1.0):
                continue
            required = _required_to_take(tgt, eta, me) + SAFETY
            if required > surplus[sid]:
                continue
            # Contest: enemy ships heading to the same target ~same time
            contest = 0
            for f in fleets:
                if int(f[1]) == me:
                    continue
                # rough: only count if enemy is moving generally toward tgt
                fdx, fdy = math.cos(f[4]), math.sin(f[4])
                to_tgt = (arr_x - f[2], arr_y - f[3])
                norm = math.hypot(*to_tgt) or 1.0
                if (fdx * to_tgt[0] + fdy * to_tgt[1]) / norm > 0.7:
                    contest += int(f[6])
            prod_val = float(tprod)
            neutral_bonus = W_NEUTRAL if int(towner) == -1 else 0.0
            home_pen = W_HOMEWARD * (math.hypot(arr_x - cmx, arr_y - cmy) / BOARD)
            score = (
                W_PROD * prod_val
                - W_ETA * eta
                - W_COST * required
                - W_CONTEST * contest
                + neutral_bonus
                - home_pen
            )
            angle = math.atan2(arr_y - sy, arr_x - sx)
            candidates.append((score, sid, int(tid), required, angle))

    candidates.sort(key=lambda c: c[0], reverse=True)

    actions = []
    for score, sid, tid, ships, angle in candidates:
        if score <= 0:
            break
        if surplus[sid] < ships:
            continue
        # Don't send tiny forces against strong production targets repeatedly.
        actions.append([sid, float(angle), int(ships)])
        surplus[sid] -= ships
        if sum(surplus.values()) < 2:
            break

    return actions
