"""v0 random bot: legal random launches. Submission plumbing sanity check."""
from __future__ import annotations

import math
import random


def agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)

    actions = []
    for p in planets:
        pid, owner, x, y, _radius, ships, _prod = p
        if owner != me or ships < 2:
            continue
        # Launch with 30% probability per owned planet, sending ~half the garrison.
        if random.random() < 0.3:
            num = max(1, int(ships // 2))
            angle = random.uniform(0.0, 2.0 * math.pi)
            actions.append([int(pid), float(angle), int(num)])
    return actions
