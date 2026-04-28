"""Unit tests for rules helpers."""
from __future__ import annotations

import math

import pytest

from rules import (
    CENTER,
    MAX_SHIP_SPEED,
    arrival_turn,
    fleet_speed,
    is_orbiting,
    predict_planet_pos,
    resolve_combat,
    score_by_owner,
    segment_hits_circle,
)


# ---------------- fleet_speed ----------------
def test_fleet_speed_bounds():
    assert fleet_speed(1) == pytest.approx(1.0)
    assert fleet_speed(0) == pytest.approx(1.0)
    v1000 = fleet_speed(1000)
    assert v1000 == pytest.approx(MAX_SHIP_SPEED, rel=1e-6)
    assert fleet_speed(10_000) == pytest.approx(MAX_SHIP_SPEED)


def test_fleet_speed_monotonic():
    prev = 0.0
    for n in (1, 2, 5, 10, 50, 100, 500, 1000):
        s = fleet_speed(n)
        assert s >= prev
        assert 1.0 <= s <= MAX_SHIP_SPEED + 1e-9
        prev = s


def test_fleet_speed_500_near_5():
    # Spec claims ~500 ships -> ~5 units/turn.
    assert 4.5 < fleet_speed(500) < 5.5


# ---------------- planet orbit prediction ----------------
def _planet(pid, x, y, r=2.0):
    return [pid, -1, x, y, r, 0, 1]


def test_is_orbiting_center_vs_edge():
    # Planet near center with small radius -> orbiting.
    assert is_orbiting(_planet(0, 55, 50, r=2))
    # Planet far from center -> orbital_r(50) + r(2) = 52 >= 50 -> static.
    assert not is_orbiting(_planet(1, 100, 50, r=2))


def test_orbit_round_trip():
    p0 = _planet(0, 70.0, 50.0, r=1.0)  # orbital_r = 20, 20+1 < 50 -> orbiting
    av = 0.05
    # After exactly 2*pi / av steps we should return near start.
    # Use exact multiple: 2*pi/0.05 = 125.66..  nearest int = 126 gives small residual.
    steps = int(round(2 * math.pi / av))
    x, y = predict_planet_pos(p0, av, steps)
    # Residual angle: 126*0.05 - 2*pi ~ 0.0168 rad, at r=20 -> ~0.34 units.
    assert math.hypot(x - 70.0, y - 50.0) < 0.5


def test_static_planet_does_not_move():
    # orbital_r = 50, planet_r = 1 -> 51 >= 50 -> static.
    p = _planet(0, 100.0, 50.0, r=1.0)
    assert predict_planet_pos(p, 0.05, 100) == (100.0, 50.0)


# ---------------- segment collision ----------------
def test_segment_hits_sun():
    assert segment_hits_circle((0.0, 50.0), (100.0, 50.0), CENTER, 10.0)


def test_segment_misses_sun():
    assert not segment_hits_circle((0.0, 0.0), (100.0, 0.0), CENTER, 10.0)


def test_segment_zero_length():
    assert segment_hits_circle((50.0, 50.0), (50.0, 50.0), CENTER, 10.0)
    assert not segment_hits_circle((0.0, 0.0), (0.0, 0.0), CENTER, 10.0)


# ---------------- arrival_turn ----------------
def test_arrival_turn_static_target():
    # Fleet of 1 ship (speed 1) travelling 10 units -> 10 turns.
    eta = arrival_turn(
        from_xy=(0.0, 0.0),
        ships=1,
        target_initial=None,
        target_xy_static=(10.0, 0.0),
        angular_velocity=0.0,
        current_step=0,
    )
    assert eta == 10


def test_arrival_turn_moving_target_converges():
    p0 = _planet(7, 70.0, 50.0, r=1.0)
    eta = arrival_turn(
        from_xy=(95.0, 50.0),
        ships=10,
        target_initial=p0,
        target_xy_static=None,
        angular_velocity=0.03,
        current_step=0,
    )
    assert 1 <= eta <= 200


# ---------------- combat ----------------
def test_combat_reinforce_own_planet():
    r = resolve_combat([(0, 5)], garrison=3, planet_owner=0)
    assert r.owner == 0 and r.ships == 8


def test_combat_capture_when_exceeds_garrison():
    r = resolve_combat([(1, 10)], garrison=3, planet_owner=0)
    assert r.owner == 1 and r.ships == 7


def test_combat_tie_top_destroys_all():
    r = resolve_combat([(1, 10), (2, 10)], garrison=5, planet_owner=0)
    assert r.owner == 0 and r.ships == 5 and r.survivors == 0


def test_combat_largest_vs_second_then_garrison():
    # Player 1 sends 10, player 2 sends 6 -> 4 surviving attackers for p1.
    # Planet garrisoned by p0 with 3 ships -> p1 flips, garrison 4-3=1.
    r = resolve_combat([(1, 10), (2, 6)], garrison=3, planet_owner=0)
    assert r.owner == 1 and r.ships == 1


def test_combat_equal_to_garrison_wipes_both():
    r = resolve_combat([(1, 5)], garrison=5, planet_owner=0)
    assert r.owner == 0 and r.ships == 0


def test_combat_empty():
    r = resolve_combat([], garrison=4, planet_owner=2)
    assert r.owner == 2 and r.ships == 4


# ---------------- scoring ----------------
def test_score_by_owner():
    planets = [
        [0, 0, 10, 10, 2.0, 5, 1],
        [1, 1, 90, 90, 2.0, 7, 1],
        [2, -1, 50, 20, 2.0, 3, 1],
    ]
    fleets = [
        [100, 0, 20, 20, 0.0, 0, 4],
        [101, 1, 80, 80, 0.0, 1, 6],
    ]
    s = score_by_owner(planets, fleets)
    assert s[0] == 9 and s[1] == 13 and s[-1] == 3
