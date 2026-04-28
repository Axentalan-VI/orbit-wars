"""v10 rank-aware control bot for Orbit Wars.

Based on v7 (Kaggle 636.1 = current best). Two surgical additions:

  A. Capital-starvation — each opponent's highest-producing planet is
     their economic "capital". Attacks on capitals get a ROI bonus.
     Even failed attempts force the opponent to defend, draining
     their expansion capacity.

  C. Rank-aware kingmaker avoidance — replaces v7's weakest-opp logic.
     We rank ourselves vs opponents by total power (planets + fleets).
     - Targets ranked ABOVE us get a bonus (catch up).
     - Targets ranked FAR BELOW us get a penalty (don't kingmake).
     - Same rank / one below: neutral.
     v7's `weakest_opp ×1.15` made us attack the trailing player, which
     in 4p FFA often helps 2nd/3rd more than us (kingmaker problem).

Everything else (ledger, predictor, defense) is byte-identical to v7.
"""

import math

BOARD = 100.0
SUN_R = 10.0
MAX_SPD = 6.0
ROT_LIM = 50.0
H_DEF = 30
H_ATK = 180
COMET_SPD = 4.0
COMET_SPAWNS = (50, 150, 250, 350, 450)

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


def _pred_target_pos(tgt_id, ini, comet_tracks, av, step_off):
    ct = comet_tracks.get(tgt_id)
    if ct is not None:
        idx, path = ct
        k = idx + step_off
        if 0 <= k < len(path):
            p = path[k]
            return float(p[0]), float(p[1])
        return None
    ip = ini.get(tgt_id)
    if ip is None:
        return None
    return _pos(ip[2], ip[3], ip[4], av, step_off)


def _arr_eta(sx, sy, ships, get_pos, start_step):
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
        cur_pos = _pred_target_pos(pid, ini, comet_tracks, av, step + 1)
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
            tpos = _pred_target_pos(pid, ini, comet_tracks, av, step + t)
            if tpos is None:
                break
            if math.hypot(nx - tpos[0], ny - tpos[1]) <= pr + 0.5:
                best_t = t
                best_pid = pid
                break

    if best_pid is None:
        return None, max_t
    return best_pid, best_t


def agent(obs, config=None):
    _init_game(obs)
    P = obs.get("planets") or []
    F = obs.get("fleets") or []
    me = int(obs.get("player", 0))
    av = float(obs.get("angular_velocity", 0))
    step = int(obs.get("step", 0))
    ini = _g.get("ini", {})
    comet_tracks = _build_comet_tracks(obs)

    my = [p for p in P if int(p[1]) == me]
    if not my:
        return []

    turns_left = max(1, 500 - step)
    early = step < 80
    late = step > 380

    # ----- 1. Arrival ledger -----
    arrivals = {int(p[0]): [] for p in P}
    for f in F:
        pid, t = _predict_fleet_target(f, P, ini, comet_tracks, av, step)
        if pid is None:
            continue
        arrivals[pid].append((t, int(f[1]), int(f[6])))
    for pid in arrivals:
        arrivals[pid].sort()

    # ----- 2. Opponent strengths + ranking + capitals -----
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

    my_ships = (sum(int(p[5]) for p in my)
                + sum(int(f[6]) for f in F if int(f[1]) == me))
    en_ships = sum(opp_power.values())

    # C: Rank players by total power (higher = better rank).
    # rank_of[owner] = 1 (leader) .. N (last). We compute our rank and
    # each opponent's rank to drive kingmaker-aware bonuses.
    all_power = dict(opp_power)
    all_power[me] = my_ships
    # sort descending by power; stable
    sorted_owners = sorted(all_power.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {}
    for i, (ow, _pw) in enumerate(sorted_owners):
        rank_of[ow] = i + 1
    my_rank = rank_of.get(me, 1)

    # A: Capital = opponent's highest-producing planet. Ties broken by
    # larger current garrison (harder to take = more strategic value).
    capital = {}  # owner -> planet_id
    best_prod = {}  # owner -> (prod, garrison)
    for p in P:
        ow = int(p[1])
        if ow < 0 or ow == me:
            continue
        prod = int(p[6])
        garr = int(p[5])
        prev = best_prod.get(ow)
        if prev is None or prod > prev[0] or (prod == prev[0] and garr > prev[1]):
            best_prod[ow] = (prod, garr)
            capital[ow] = int(p[0])

    # ----- 3. Defensive ledger -----
    doomed = set()
    reserve = {}
    for p in my:
        pid = int(p[0])
        g0, prod = int(p[5]), int(p[6])
        arr = arrivals.get(pid, [])
        my_cum = 0
        en_cum = 0
        max_def = 0
        earliest_loss = None
        owner_now = me
        garrison = g0
        prev_t = 0
        for (t, ow, s) in arr:
            if t > H_DEF:
                break
            if owner_now == me:
                garrison += prod * (t - prev_t)
            elif owner_now >= 0:
                garrison += prod * (t - prev_t)
            if ow == me:
                my_cum += s
                if owner_now == me:
                    garrison += s
                else:
                    if s > garrison:
                        owner_now = me
                        garrison = s - garrison
                    else:
                        garrison = garrison - s
            else:
                en_cum += s
                if owner_now == me:
                    if s > garrison:
                        owner_now = ow
                        garrison = s - garrison
                        earliest_loss = t if earliest_loss is None else earliest_loss
                    else:
                        garrison = garrison - s
                        max_def = max(max_def, s)
                else:
                    pass
            prev_t = t
            max_def = max(max_def, en_cum - my_cum - g0 - prod * t)
        if earliest_loss is not None and en_cum > (g0 + prod * earliest_loss + my_cum) * 1.3:
            doomed.add(pid)
            reserve[pid] = 0
        else:
            need = max(0, max_def + 2)
            reserve[pid] = min(g0, need if need > 0 else 1)
    surplus = {int(p[0]): max(0, int(p[5]) - reserve[int(p[0])]) for p in my}

    # ----- 4. Weighted center of mass -----
    tp = sum(int(p[6]) for p in my) or 1
    cmx = sum(float(p[2]) * int(p[6]) for p in my) / tp
    cmy = sum(float(p[3]) * int(p[6]) for p in my) / tp

    # ----- 5. Required-ships computation -----
    cids = set(int(c) for c in (obs.get("comet_planet_ids") or []))

    comet_reserve_active = any(0 < s - step <= 10 for s in COMET_SPAWNS)

    def required_for(tgt, eta):
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
            return 0, owner
        return garrison + 1, owner

    def _post_capture_threat(tid, our_eta, our_ships, tgt):
        tprod = max(1, int(tgt[6]))
        arr = arrivals.get(tid, [])
        threat = 0
        for (t, ow, s) in arr:
            if t <= our_eta:
                continue
            if t > our_eta + 5:
                break
            if ow != me and ow != -1:
                threat += s
        return threat

    # ----- 6. Score all (source, target) pairs -----
    cands = []
    for tgt in P:
        tid, tow = int(tgt[0]), int(tgt[1])
        if tow == me:
            continue
        tprod = max(1, int(tgt[6]))
        is_comet = tid in cids

        if tow == -1:
            ow_sums = {}
            for (t, ow, s) in arrivals.get(tid, []):
                if t <= 25 and ow != me and ow != -1:
                    ow_sums[ow] = ow_sums.get(ow, 0) + s
            if len(ow_sums) >= 2:
                vals = sorted(ow_sums.values(), reverse=True)
                if vals[1] >= 0.7 * vals[0]:
                    continue

        if tow == -1 and sum(s for (_t, o, s) in arrivals.get(tid, []) if o == me) > \
                int(tgt[5]) + 3 and not any(o != me and o != -1 for (_t, o, _s) in arrivals.get(tid, [])):
            continue

        for src in my:
            sid = int(src[0])
            if surplus[sid] < 2:
                continue
            sx, sy = float(src[2]), float(src[3])

            def get_pos(off):
                return _pred_target_pos(tid, ini, comet_tracks, av, step + off)

            eta1, ax1, ay1 = _arr_eta(sx, sy, surplus[sid], get_pos, step)
            if eta1 is None or eta1 > H_ATK:
                continue
            if late and eta1 > 25:
                continue
            ct = comet_tracks.get(tid)
            if ct is not None:
                remaining = len(ct[1]) - ct[0] - 1
                if eta1 > remaining - 3:
                    continue

            angle, ax1, ay1 = _best_angle(sx, sy, ax1, ay1)
            if angle is None:
                continue

            req, future_owner = required_for(tgt, eta1)
            if req == 0:
                continue
            ships = req + 1
            if ships > surplus[sid]:
                continue

            eta2, ax2, ay2 = _arr_eta(sx, sy, ships, get_pos, step)
            if eta2 is None:
                continue
            if ct is not None:
                remaining = len(ct[1]) - ct[0] - 1
                if eta2 > remaining - 3:
                    continue
            angle2, ax2, ay2 = _best_angle(sx, sy, ax2, ay2)
            if angle2 is None:
                continue
            req2, fo2 = required_for(tgt, eta2)
            if req2 == 0:
                continue
            ships2 = req2 + 1

            threat = _post_capture_threat(tid, eta2, ships2, tgt)
            if threat > 0:
                ships2 += threat

            if ships2 > surplus[sid]:
                continue

            prod_gain = tprod * max(0, turns_left - eta2)
            roi = prod_gain / ships2

            if tow == -1:
                roi *= 2.0 if early else 1.3
            if is_comet:
                roi *= 1.6

            roi -= math.hypot(ax2 - cmx, ay2 - cmy) * 0.025

            # --- C: Rank-aware kingmaker avoidance ---
            # Replaces v7's weakest_opp ±15%/×0.9 block.
            # Attack players ranked at or above us; avoid kingmaking
            # by heavily attacking players ranked much below us.
            if is_4p and tow >= 0:
                their_rank = rank_of.get(tow, my_rank)
                # rank_gap > 0 means target is ranked BELOW us (worse).
                # rank_gap < 0 means target is ranked ABOVE us (better).
                rank_gap = their_rank - my_rank
                if rank_gap < 0:
                    # they're above us — catch up
                    roi *= 1.20
                elif rank_gap == 0:
                    # same rank (can't happen in 4p with unique ranks
                    # but safe)
                    roi *= 1.00
                elif rank_gap == 1:
                    # one slot below — neutral, normal competition
                    roi *= 1.00
                else:
                    # 2+ slots below — don't kingmake
                    roi *= 0.80

            # --- A: Capital-starvation bonus ---
            # Attacking an opponent's highest-producing planet is
            # strategically valuable: even failed attacks force
            # defensive commitments that slow their expansion.
            if tow >= 0 and tow != me and capital.get(tow) == tid:
                roi *= 1.30

            if en_ships > my_ships * 1.3 and tow >= 0 and tow != me:
                roi *= 1.3

            if sid in doomed:
                roi += 10.0

            if comet_reserve_active and not is_comet and tow == -1 and ships2 > 15:
                roi *= 0.7

            cands.append((roi, sid, tid, ships2, angle2, eta2))

    cands.sort(key=lambda c: c[0], reverse=True)

    # ----- 7. Greedy assignment with wave stacking -----
    actions = []
    target_committed = {}
    for roi, sid, tid, ships, angle, eta in cands:
        if roi <= 0:
            break
        if surplus[sid] < ships:
            continue
        prior = target_committed.get(tid, [])
        if prior:
            total_prior = sum(s for _e, s in prior)
            if total_prior > 150:
                continue
        actions.append([sid, float(angle), int(ships)])
        surplus[sid] -= ships
        target_committed.setdefault(tid, []).append((eta, ships))
        if sum(surplus.values()) < 2:
            break

    # ----- 8. Timed reinforcement of threatened allies -----
    if not late:
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
                if sid == tid or surplus[sid] < 4:
                    continue
                if arrivals.get(sid) and any(o != me for (_t, o, _s)
                                              in arrivals[sid][:3]):
                    continue
                sx, sy = float(src[2]), float(src[3])

                def get_pos(off, tid=tid):
                    return _pred_target_pos(tid, ini, comet_tracks, av, step + off)

                eta_r, rx, ry = _arr_eta(sx, sy, surplus[sid], get_pos, step)
                if eta_r is None or eta_r >= earliest_en:
                    continue
                ang, rx, ry = _best_angle(sx, sy, rx, ry)
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
