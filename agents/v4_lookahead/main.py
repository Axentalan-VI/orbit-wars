
import math
import time

BOARD = 100.0
SUN_R = 10.0
MAX_SPD = 6.0
ROT_LIM = 50.0
H_DEF = 30
H_ATK = 180
H_EVAL = 40  # lookahead horizon for scoring candidate profiles
COMET_SPAWNS = (50, 150, 250, 350, 450)

# Default scoring profiles (can be overridden via PROFILES_OVERRIDE for tuning)
DEFAULT_PROFILES = [
    # Balanced (v3-style)
    dict(comet_mul=1.6, neutral_mul=1.0, aggression_mul=1.3,
         safety_margin=1, weakest_opp_mul=1.15, other_opp_mul=0.9,
         compact_pen=0.025),
    # Aggressive
    dict(comet_mul=1.4, neutral_mul=0.9, aggression_mul=1.5,
         safety_margin=1, weakest_opp_mul=1.25, other_opp_mul=0.95,
         compact_pen=0.01),
    # Defensive / compact
    dict(comet_mul=1.6, neutral_mul=1.1, aggression_mul=1.1,
         safety_margin=2, weakest_opp_mul=1.1, other_opp_mul=0.8,
         compact_pen=0.05),
    # Comet-heavy
    dict(comet_mul=2.2, neutral_mul=1.1, aggression_mul=1.2,
         safety_margin=1, weakest_opp_mul=1.15, other_opp_mul=0.9,
         compact_pen=0.025),
]
PROFILES_OVERRIDE = None  # set by tuner to a list[dict]

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
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _pred_target_pos(tid, ini, comet_tracks, av, step_off):
    ct = comet_tracks.get(tid)
    if ct is not None:
        idx, path = ct
        k = idx + step_off
        if 0 <= k < len(path):
            p = path[k]
            return float(p[0]), float(p[1])
        return None
    ip = ini.get(tid)
    if ip is None:
        return None
    return _pos(ip[2], ip[3], ip[4], av, step_off)


def _arr_eta(sx, sy, ships, get_pos):
    sp = _spd(ships)
    init = get_pos(1)
    if init is None:
        return None, None, None
    tx, ty = init
    eta = 1
    for _ in range(6):
        d = math.hypot(sx - tx, sy - ty)
        neta = max(1, int(math.ceil(d / sp)))
        npos = get_pos(neta)
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


def _init_game(obs):
    ini = obs.get("initial_planets") or []
    key = tuple((int(p[0]), round(float(p[2]), 2), round(float(p[3]), 2))
                for p in ini[:6])
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


# ---------- B1: better fleet-target classifier ----------
def _predict_fleet_target(f, P, ini, comet_tracks, av, step, max_t=60):
    """Planet with minimum (closest-approach distance / planet_radius).

    Also returns (pid, eta). Falls back to first-hit if nothing is within
    radius at any step (fleet might miss or go out of bounds).
    """
    fx, fy, fa, fs = float(f[2]), float(f[3]), float(f[4]), int(f[6])
    sp = _spd(fs)
    vx, vy = math.cos(fa) * sp, math.sin(fa) * sp

    best_pid, best_t, best_score = None, None, 1e9
    for p in P:
        pid, pr = int(p[0]), float(p[4])
        # reject: opposing direction
        cur = _pred_target_pos(pid, ini, comet_tracks, av, step + 1)
        if cur is None:
            continue
        if vx * (cur[0] - fx) + vy * (cur[1] - fy) <= 0:
            continue
        # scan for closest approach
        best_d = 1e9
        best_k = None
        for t in range(1, max_t):
            nx, ny = fx + vx * t, fy + vy * t
            if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                break
            tp = _pred_target_pos(pid, ini, comet_tracks, av, step + t)
            if tp is None:
                break
            d = math.hypot(nx - tp[0], ny - tp[1])
            if d <= pr + 0.5:   # definite hit
                best_d = d
                best_k = t
                break
            if d < best_d:
                best_d = d
                best_k = t
            # pruning: if distance growing substantially, stop
            if t > 5 and d > 3 * pr + 5:
                break
        if best_k is not None and best_d < pr + 2.0:
            score = best_d / max(pr, 0.5)
            if score < best_score:
                best_score = score
                best_pid = pid
                best_t = best_k
    return best_pid, best_t


def _opp_arrivals_summary(arrivals, me, tid, horizon):
    """For a target, return list of (eta, owner, ships) for each non-me owner."""
    per = {}
    for (t, ow, s) in arrivals.get(tid, []):
        if t > horizon or ow == me:
            continue
        per.setdefault(ow, []).append((t, s))
    return per


# ---------- Arrival-ledger simulation of one planet to horizon ----------
def _simulate_planet(tgt_init, arrivals_for, hypothetical, horizon):
    """Replay arrivals on this planet to `horizon` turns ahead.

    `tgt_init` = (tid, owner, x, y, r, ships, prod) snapshot NOW.
    `arrivals_for` = sorted list of (eta, owner, ships) real in-flight fleets.
    `hypothetical` = sorted list of (eta, owner, ships) new fleets we may launch.
    Returns (final_owner, final_ships).
    """
    _tid, owner, _x, _y, _r, ships, prod = tgt_init
    prod = max(1, int(prod))
    garrison = int(ships)
    # merge and iterate in time order
    events = sorted(list(arrivals_for) + list(hypothetical))
    prev_t = 0
    for (t, ow, s) in events:
        if t > horizon:
            break
        if owner >= 0:
            garrison += prod * (t - prev_t)
        # Simplified combat: single attacker vs garrison
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
        garrison += prod * (horizon - prev_t)
    return owner, garrison


def _evaluate_actions(actions, ctx):
    """Score an action set: predicted (my_ships - strongest_enemy) at T+H_EVAL."""
    me = ctx["me"]
    P = ctx["P"]
    arrivals = ctx["arrivals"]
    ini = ctx["ini"]
    comet_tracks = ctx["comet_tracks"]
    av = ctx["av"]
    step = ctx["step"]
    step_surplus = dict(ctx["surplus"])    # copy: we'll deduct

    # Build hypothetical arrivals from actions
    hyp_per_target = {}
    for (sid, ang, ships) in actions:
        # find source planet
        src = ctx["my_by_id"].get(sid)
        if src is None:
            continue
        if ships > step_surplus.get(sid, 0):
            continue
        step_surplus[sid] -= ships
        sx, sy = float(src[2]), float(src[3])
        # find which target this angle heads to
        sp = _spd(ships)
        vx, vy = math.cos(ang) * sp, math.sin(ang) * sp
        best_tid, best_t = None, None
        best_d = 1e9
        for p in P:
            pid, pr = int(p[0]), float(p[4])
            if pid == sid:
                continue
            cur = _pred_target_pos(pid, ini, comet_tracks, av, step + 1)
            if cur is None:
                continue
            if vx * (cur[0] - sx) + vy * (cur[1] - sy) <= 0:
                continue
            d_best = 1e9
            k_best = None
            for t in range(1, 80):
                nx, ny = sx + vx * t, sy + vy * t
                if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
                    break
                tp = _pred_target_pos(pid, ini, comet_tracks, av, step + t)
                if tp is None:
                    break
                d = math.hypot(nx - tp[0], ny - tp[1])
                if d <= pr + 0.5:
                    d_best, k_best = d, t
                    break
                if d < d_best:
                    d_best, k_best = d, t
                if t > 5 and d > 3 * pr + 5:
                    break
            if k_best is not None and d_best < pr + 2.0:
                score = d_best / max(pr, 0.5)
                if score < best_d:
                    best_d, best_tid, best_t = score, pid, k_best
        if best_tid is not None:
            hyp_per_target.setdefault(best_tid, []).append((best_t, me, ships))

    # Simulate every planet forward H_EVAL turns
    my_total = 0
    opp_total = {}
    for p in P:
        pid = int(p[0])
        arr = arrivals.get(pid, [])
        hyp = hyp_per_target.get(pid, [])
        own, ships = _simulate_planet(p, arr, hyp, H_EVAL)
        if own == me:
            my_total += ships
        elif own >= 0:
            opp_total[own] = opp_total.get(own, 0) + ships
    # Add ships still in flight (we don't remove real/hyp fleets that arrive
    # AFTER H_EVAL, so count them separately).
    for f in ctx["F"]:
        if int(f[1]) == me:
            my_total += int(f[6])
        else:
            ow = int(f[1])
            opp_total[ow] = opp_total.get(ow, 0) + int(f[6])
    strongest_opp = max(opp_total.values()) if opp_total else 0
    return my_total - strongest_opp, my_total


# ---------- candidate action generator ----------
def _generate_candidates(ctx, profile):
    """Generate a candidate action list under a given scoring profile.

    profile dict keys: 'comet_mul', 'neutral_mul', 'aggression_mul',
    'safety_margin', 'weakest_opp_mul', 'other_opp_mul', 'compact_pen'
    """
    me = ctx["me"]
    P = ctx["P"]
    F = ctx["F"]
    av = ctx["av"]
    step = ctx["step"]
    ini = ctx["ini"]
    comet_tracks = ctx["comet_tracks"]
    arrivals = ctx["arrivals"]
    surplus = dict(ctx["surplus"])
    cmx, cmy = ctx["cmx"], ctx["cmy"]
    turns_left = ctx["turns_left"]
    early, late = ctx["early"], ctx["late"]
    is_4p, weakest_opp = ctx["is_4p"], ctx["weakest_opp"]
    my_ships, en_ships = ctx["my_ships"], ctx["en_ships"]
    doomed = ctx["doomed"]
    cids = ctx["cids"]
    comet_reserve_active = ctx["comet_reserve_active"]

    safety = profile["safety_margin"]
    comet_mul = profile["comet_mul"]
    neutral_mul = profile["neutral_mul"]
    aggression_mul = profile["aggression_mul"]
    weakest_mul = profile["weakest_opp_mul"]
    other_mul = profile["other_opp_mul"]
    compact_pen = profile["compact_pen"]

    cands = []
    for tgt in P:
        tid, tow = int(tgt[0]), int(tgt[1])
        if tow == me:
            continue
        tprod = max(1, int(tgt[6]))
        is_comet = tid in cids

        # --- D3: if two enemies will converge here, let them fight ---
        if tow == -1:
            ow_sums = {}
            for (t, ow, s) in arrivals.get(tid, []):
                if t <= 25 and ow != me and ow != -1:
                    ow_sums[ow] = ow_sums.get(ow, 0) + s
            if len(ow_sums) >= 2:
                vals = sorted(ow_sums.values(), reverse=True)
                # top two within 30% of each other -> mostly annihilate
                if vals[1] >= 0.7 * vals[0]:
                    continue

        # already captured by prior mine?
        my_inbound = sum(s for (_t, o, _s) in arrivals.get(tid, [])
                         if o == me for s in [_s])
        # Simpler version:
        my_inbound = sum(s for (_t, o, s) in arrivals.get(tid, []) if o == me)
        if tow == -1 and my_inbound > int(tgt[5]) + 3 and not any(
                o != me and o != -1 for (_t, o, _s) in arrivals.get(tid, [])):
            continue

        for src in ctx["my"]:
            sid = int(src[0])
            if surplus.get(sid, 0) < 2:
                continue
            sx, sy = float(src[2]), float(src[3])

            def get_pos(off, tid=tid):
                return _pred_target_pos(tid, ini, comet_tracks, av, step + off)

            eta1, ax, ay = _arr_eta(sx, sy, surplus[sid], get_pos)
            if eta1 is None or eta1 > H_ATK:
                continue
            if late and eta1 > 25:
                continue
            ct = comet_tracks.get(tid)
            if ct is not None and eta1 > len(ct[1]) - ct[0] - 4:
                continue
            angle, ax, ay = _best_angle(sx, sy, ax, ay)
            if angle is None:
                continue

            req = _required_at(tgt, eta1, arrivals, me)
            if req == 0:
                continue
            ships = req + safety
            if ships > surplus[sid]:
                continue
            # refine with actual fleet size
            eta2, ax2, ay2 = _arr_eta(sx, sy, ships, get_pos)
            if eta2 is None:
                continue
            if ct is not None and eta2 > len(ct[1]) - ct[0] - 4:
                continue
            a2, ax2, ay2 = _best_angle(sx, sy, ax2, ay2)
            if a2 is None:
                continue
            req2 = _required_at(tgt, eta2, arrivals, me)
            if req2 == 0:
                continue
            ships2 = req2 + safety
            if ships2 > surplus[sid]:
                continue

            prod_gain = tprod * max(0, turns_left - eta2)
            roi = prod_gain / ships2

            if tow == -1:
                roi *= (2.0 * neutral_mul) if early else (1.3 * neutral_mul)
            if is_comet:
                roi *= comet_mul

            # Compactness penalty
            roi -= math.hypot(ax2 - cmx, ay2 - cmy) * compact_pen

            if is_4p and tow >= 0:
                roi *= (weakest_mul if tow == weakest_opp else other_mul)

            if en_ships > my_ships * 1.3 and tow >= 0 and tow != me:
                roi *= aggression_mul

            if sid in doomed:
                roi += 10.0

            # E2: comet-spawn reserve window — discourage expensive attacks
            if comet_reserve_active and not is_comet and tow == -1 and ships2 > 15:
                roi *= 0.7

            cands.append((roi, sid, tid, ships2, a2, eta2))

    cands.sort(key=lambda c: c[0], reverse=True)

    actions = []
    committed = {}  # tid -> total ships launched this turn (wave stacking cap)
    for roi, sid, tid, ships, ang, eta in cands:
        if roi <= 0:
            break
        if surplus[sid] < ships:
            continue
        # Cap cumulative commitment per target (v3-style wave stacking)
        if committed.get(tid, 0) > 150:
            continue
        actions.append([sid, float(ang), int(ships)])
        surplus[sid] -= ships
        committed[tid] = committed.get(tid, 0) + ships
        if sum(surplus.values()) < 2:
            break
    # reinforcement is done OUTSIDE candidate evaluation (not profile-dependent)
    return actions, surplus


def _required_at(tgt, eta, arrivals, me):
    """Ships needed to flip tgt at turn eta (result > 0 → need that many)."""
    tid, owner = int(tgt[0]), int(tgt[1])
    tprod = max(1, int(tgt[6]))
    garrison = int(tgt[5])
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


def agent(obs, config=None):
    t_start = time.time()
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

    turns_left = max(1, 500 - step)
    early = step < 80
    late = step > 380

    # arrivals ledger
    arrivals = {int(p[0]): [] for p in P}
    for f in F:
        pid, t = _predict_fleet_target(f, P, ini, comet_tracks, av, step)
        if pid is None:
            continue
        arrivals[pid].append((t, int(f[1]), int(f[6])))
    for pid in arrivals:
        arrivals[pid].sort()

    # opponent powers
    opp_power = {}
    for p in P:
        ow = int(p[1])
        if ow >= 0 and ow != me:
            opp_power[ow] = opp_power.get(ow, 0) + int(p[5])
    for f in F:
        ow = int(f[1])
        if ow >= 0 and ow != me:
            opp_power[ow] = opp_power.get(ow, 0) + int(f[6])
    is_4p = len(opp_power) >= 2
    weakest_opp = min(opp_power, key=opp_power.get) if opp_power else None
    my_ships = (sum(int(p[5]) for p in my)
                + sum(int(f[6]) for f in F if int(f[1]) == me))
    en_ships = sum(opp_power.values())

    # defensive reserve (same as v3)
    doomed = set()
    reserve = {}
    for p in my:
        pid = int(p[0])
        g0, prod = int(p[5]), int(p[6])
        arr = arrivals.get(pid, [])
        my_cum = en_cum = max_def = 0
        owner_now = me
        garrison = g0
        prev_t = 0
        earliest_loss = None
        for (t, ow, s) in arr:
            if t > H_DEF:
                break
            if owner_now >= 0:
                garrison += prod * (t - prev_t)
            if ow == me:
                my_cum += s
                if owner_now == me:
                    garrison += s
                else:
                    if s > garrison:
                        owner_now, garrison = me, s - garrison
                    else:
                        garrison -= s
            else:
                en_cum += s
                if owner_now == me:
                    if s > garrison:
                        owner_now, garrison = ow, s - garrison
                        if earliest_loss is None:
                            earliest_loss = t
                    else:
                        garrison -= s
                        max_def = max(max_def, s)
            prev_t = t
            max_def = max(max_def, en_cum - my_cum - g0 - prod * t)
        if earliest_loss is not None and en_cum > (g0 + prod * earliest_loss + my_cum) * 1.3:
            doomed.add(pid)
            reserve[pid] = 0
        else:
            need = max(0, max_def + 2)
            reserve[pid] = min(g0, need if need > 0 else 1)
    surplus = {int(p[0]): max(0, int(p[5]) - reserve[int(p[0])]) for p in my}

    # weighted center
    tp = sum(int(p[6]) for p in my) or 1
    cmx = sum(float(p[2]) * int(p[6]) for p in my) / tp
    cmy = sum(float(p[3]) * int(p[6]) for p in my) / tp

    # E2: comet-spawn reserve — active in last 10 turns before a spawn
    comet_reserve_active = any(0 < s - step <= 10 for s in COMET_SPAWNS)

    ctx = dict(
        me=me, P=P, F=F, my=my,
        my_by_id={int(p[0]): p for p in my},
        av=av, step=step, ini=ini, comet_tracks=comet_tracks,
        arrivals=arrivals, surplus=surplus, cmx=cmx, cmy=cmy,
        turns_left=turns_left, early=early, late=late,
        is_4p=is_4p, weakest_opp=weakest_opp,
        my_ships=my_ships, en_ships=en_ships,
        doomed=doomed, cids=cids,
        comet_reserve_active=comet_reserve_active,
    )

    # Candidate profiles
    profiles = PROFILES_OVERRIDE if PROFILES_OVERRIDE is not None else DEFAULT_PROFILES

    # Time budget: don't eval if we're running out of time
    BUDGET = 0.55  # seconds of total work allowed
    best_actions, best_score = None, -1e18
    for i, prof in enumerate(profiles):
        if time.time() - t_start > BUDGET:
            break
        actions_i, surplus_i = _generate_candidates(ctx, prof)
        score, _my = _evaluate_actions(actions_i, ctx)
        if score > best_score:
            best_score = score
            best_actions = actions_i
            best_surplus = surplus_i

    if best_actions is None:
        # fallback: empty action
        best_actions = []
        best_surplus = surplus

    # ----- Reinforcement (from v3, post-selection, uses surplus_left) -----
    if not late:
        surplus_left = best_surplus
        for tgt in my:
            tid = int(tgt[0])
            if tid in doomed:
                continue
            t_arr = arrivals.get(tid, [])
            if not t_arr:
                continue
            enemy_total = sum(s for (_t, o, s) in t_arr if o != me and _t <= H_DEF)
            my_inbound = sum(s for (_t, o, s) in t_arr if o == me and _t <= H_DEF)
            garrison = int(tgt[5]) + my_inbound
            if enemy_total <= garrison + int(tgt[6]) * H_DEF:
                continue
            need = enemy_total - garrison - int(tgt[6]) * 5
            if need <= 2:
                continue
            earliest_en = min((t for (t, o, _s) in t_arr if o != me), default=H_DEF)
            donors = []
            for src in my:
                sid = int(src[0])
                if sid == tid or surplus_left.get(sid, 0) < 4:
                    continue
                if any(o != me for (_t, o, _s) in arrivals.get(sid, [])[:3]):
                    continue
                sx, sy = float(src[2]), float(src[3])

                def get_pos(off, tid=tid):
                    return _pred_target_pos(tid, ini, comet_tracks, av, step + off)

                eta_r, rx, ry = _arr_eta(sx, sy, surplus_left[sid], get_pos)
                if eta_r is None or eta_r >= earliest_en:
                    continue
                ang, rx, ry = _best_angle(sx, sy, rx, ry)
                if ang is None:
                    continue
                donors.append((eta_r, sid, ang))
            donors.sort()
            remaining = need
            for _e, sid, ang in donors:
                if remaining <= 0:
                    break
                send = min(surplus_left[sid] - 1, remaining + 2)
                if send < 3:
                    continue
                best_actions.append([sid, float(ang), int(send)])
                surplus_left[sid] -= send
                remaining -= send

    return best_actions
