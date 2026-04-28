"""Pure-Python / NumPy rules helpers for Orbit Wars.

Mirrors the Kaggle engine spec so that both the submitted bot and training code
can share logic without importing ``kaggle_environments`` at inference time.

Spec ref (2026-04-20): https://www.kaggle.com/competitions/orbit-wars
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants (match engine defaults unless overridden via config)
# ---------------------------------------------------------------------------
BOARD = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
MAX_SHIP_SPEED = 6.0
COMET_SPEED = 4.0
STEPS = 500
ROTATION_RADIUS_LIMIT = 50.0  # orbital_r + planet_r < 50 -> orbiting
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

# Planet/fleet tuple layouts (also exposed as named tuples by the engine).
PLANET_FIELDS = ("id", "owner", "x", "y", "radius", "ships", "production")
FLEET_FIELDS = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")


# ---------------------------------------------------------------------------
# Fleet speed
# ---------------------------------------------------------------------------
def fleet_speed(ships: float, max_speed: float = MAX_SHIP_SPEED) -> float:
    """Speed (units/turn) of a fleet containing ``ships`` ships.

    ``speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5``, clamped
    to ``[1, max_speed]``. 1 ship moves at 1.0, ~1000 ships approaches max.
    """
    if ships <= 1:
        return 1.0
    t = math.log(ships) / math.log(1000.0)
    # Spec uses ^1.5 but larger fleets than 1000 should not accelerate further.
    t = max(0.0, min(1.0, t))
    return 1.0 + (max_speed - 1.0) * (t ** 1.5)


# ---------------------------------------------------------------------------
# Planet / comet position prediction
# ---------------------------------------------------------------------------
def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def is_orbiting(planet: Sequence[float]) -> bool:
    """True if the planet rotates around the sun (per the spec's radius check)."""
    _pid, _owner, x, y, radius, *_ = planet
    orbital_r = _dist(x, y, *CENTER)
    return (orbital_r + radius) < ROTATION_RADIUS_LIMIT


def predict_planet_pos(
    initial_planet: Sequence[float],
    angular_velocity: float,
    step: int,
) -> tuple[float, float]:
    """Predict ``(x, y)`` at ``step`` turns after game start.

    Non-orbiting planets stay put. Orbiting planets rotate around the sun at
    ``angular_velocity`` radians/turn from their initial angular position.
    """
    _pid, _owner, x0, y0, radius, *_ = initial_planet
    if not is_orbiting(initial_planet):
        return float(x0), float(y0)
    cx, cy = CENTER
    dx, dy = x0 - cx, y0 - cy
    r = math.hypot(dx, dy)
    theta0 = math.atan2(dy, dx)
    theta = theta0 + angular_velocity * step
    return cx + r * math.cos(theta), cy + r * math.sin(theta)


def predict_comet_pos(
    comet_group: dict,
    comet_planet_id: int,
    dt: int,
) -> tuple[float, float] | None:
    """Predict a comet's position ``dt`` turns in the future.

    Returns ``None`` if the comet would have left the board (path exhausted).
    ``comet_group`` is an entry from ``obs['comets']`` containing ``paths``
    (list of per-comet trajectories) and ``path_index`` (current step index).
    """
    try:
        ids = comet_group["planet_ids"]
        idx = ids.index(comet_planet_id)
    except (KeyError, ValueError):
        return None
    path = comet_group["paths"][idx]
    cur = int(comet_group.get("path_index", 0))
    target = cur + dt
    if target < 0 or target >= len(path):
        return None
    x, y = path[target][:2]
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------
def segment_hits_circle(
    p0: tuple[float, float],
    p1: tuple[float, float],
    c: tuple[float, float],
    r: float,
) -> bool:
    """True if line segment ``p0 -> p1`` passes within ``r`` of center ``c``."""
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


# ---------------------------------------------------------------------------
# Arrival-turn estimator (moving target)
# ---------------------------------------------------------------------------
def arrival_turn(
    from_xy: tuple[float, float],
    ships: int,
    target_initial: Sequence[float] | None,
    target_xy_static: tuple[float, float] | None,
    angular_velocity: float,
    current_step: int,
    max_speed: float = MAX_SHIP_SPEED,
    max_iter: int = 4,
) -> int:
    """Estimate turns to reach a (possibly moving) target.

    If ``target_initial`` is provided (a planet tuple from ``initial_planets``),
    orbital motion is accounted for via fixed-point iteration. Otherwise
    ``target_xy_static`` is used as a fixed point (e.g. comet intercept).
    """
    speed = fleet_speed(ships, max_speed=max_speed)
    if target_xy_static is not None and target_initial is None:
        tx, ty = target_xy_static
        d = _dist(*from_xy, tx, ty)
        return max(1, math.ceil(d / speed))

    assert target_initial is not None
    # Seed with current predicted target position
    tx, ty = predict_planet_pos(target_initial, angular_velocity, current_step)
    eta = 1
    for _ in range(max_iter):
        d = _dist(*from_xy, tx, ty)
        eta = max(1, math.ceil(d / speed))
        nx, ny = predict_planet_pos(
            target_initial, angular_velocity, current_step + eta
        )
        if abs(nx - tx) + abs(ny - ty) < 1e-6:
            tx, ty = nx, ny
            break
        tx, ty = nx, ny
    return eta


# ---------------------------------------------------------------------------
# Combat resolution (matches turn-order step 7)
# ---------------------------------------------------------------------------
@dataclass
class CombatResult:
    owner: int           # new planet owner (may equal input)
    ships: int           # new garrison
    survivors_owner: int  # owner of surviving attacker force (-2 if none)
    survivors: int        # count of surviving attackers that landed


def resolve_combat(
    arrivals: Iterable[tuple[int, int]],
    garrison: int,
    planet_owner: int,
) -> CombatResult:
    """Resolve combat on a single planet.

    Steps (per spec):
    1. Group arrivals by owner, summing ships.
    2. Largest force fights 2nd largest - difference survives; a tie destroys
       all attackers.
    3. Surviving attacker vs. garrison: reinforce if same owner, else fight.
       If attackers > garrison, planet flips and garrison = surplus.
    """
    totals: dict[int, int] = {}
    for owner, ships in arrivals:
        if ships <= 0:
            continue
        totals[owner] = totals.get(owner, 0) + int(ships)

    if not totals:
        return CombatResult(planet_owner, garrison, -2, 0)

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) == 1:
        surv_owner, surv = ordered[0]
    else:
        (o1, s1), (_o2, s2) = ordered[0], ordered[1]
        if s1 == s2:
            # Tie at the top -> every attacker destroyed.
            return CombatResult(planet_owner, garrison, -2, 0)
        surv_owner, surv = o1, s1 - s2

    if surv <= 0:
        return CombatResult(planet_owner, garrison, -2, 0)

    if surv_owner == planet_owner:
        return CombatResult(planet_owner, garrison + surv, surv_owner, surv)

    if surv > garrison:
        return CombatResult(surv_owner, surv - garrison, surv_owner, surv)
    if surv == garrison:
        # All destroyed on both sides - planet becomes neutral-ish? Spec says
        # attackers need to exceed garrison to flip; equal means wiped out on
        # both sides but planet keeps prior owner.
        return CombatResult(planet_owner, 0, -2, 0)
    return CombatResult(planet_owner, garrison - surv, -2, 0)


# ---------------------------------------------------------------------------
# Convenience: ship total per owner (scoring)
# ---------------------------------------------------------------------------
def score_by_owner(
    planets: Iterable[Sequence[float]],
    fleets: Iterable[Sequence[float]],
) -> dict[int, int]:
    """Total ships (planets + fleets) per owner. Neutrals (-1) included."""
    totals: dict[int, int] = {}
    for p in planets:
        owner = int(p[1])
        ships = int(p[5])
        if ships:
            totals[owner] = totals.get(owner, 0) + ships
    for f in fleets:
        owner = int(f[1])
        ships = int(f[6])
        totals[owner] = totals.get(owner, 0) + ships
    return totals
