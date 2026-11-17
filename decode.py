"""Viterbi decode over H, A, D with a minimum length for every stoppage.

A real stoppage lasts seconds -- the shortest in the public window is 5.5 s, and even a quick
throw-in takes a few. Version 3 still produced 38 stoppages under 2.5 s across the match, each
one splitting a possession in two. Here D is a chain of MIN_DEAD states that must be walked end
to end before the decode may leave it, so a stoppage is either long enough or not there at all.
The decision is made inside the decode, with all the evidence, rather than by deleting short D
runs afterwards.

States: 0 = H, 1 = A, 2 .. 2 + MIN_DEAD - 1 = D (entered at 2, left from the last).
"""

import numpy as np

H_STATE, A_STATE = 0, 1


def viterbi_min_dead(cost, switch_team, switch_dead, min_dead):
    """cost (n, 3) per frame for H, A, D. Returns (n,) codes 0 = H, 1 = A, 2 = D."""
    n = len(cost)
    k = max(int(min_dead), 1)
    first_dead, last_dead = 2, 2 + k - 1
    total = np.full(2 + k, np.inf)
    total[H_STATE], total[A_STATE], total[first_dead] = cost[0, 0], cost[0, 1], cost[0, 2]
    if k == 1:
        total[first_dead] = cost[0, 2]
    # back[t, s] = state at t-1 on the cheapest path into s at t.
    back = np.empty((n, 2 + k), dtype=np.int16)
    chain = np.arange(first_dead, last_dead)  # D_1 .. D_{k-1}, each feeds the next

    for t in range(1, n):
        new = np.empty_like(total)
        # Into H: stay, switch from A, or leave the stoppage from its last state.
        options = (total[H_STATE], total[A_STATE] + switch_team, total[last_dead] + switch_dead)
        pick = int(np.argmin(options))
        new[H_STATE] = options[pick] + cost[t, 0]
        back[t, H_STATE] = (H_STATE, A_STATE, last_dead)[pick]

        options = (total[A_STATE], total[H_STATE] + switch_team, total[last_dead] + switch_dead)
        pick = int(np.argmin(options))
        new[A_STATE] = options[pick] + cost[t, 1]
        back[t, A_STATE] = (A_STATE, H_STATE, last_dead)[pick]

        # Into the first stoppage state, from either team.
        from_team = H_STATE if total[H_STATE] <= total[A_STATE] else A_STATE
        entry = total[from_team] + switch_dead
        if k == 1:
            stay = total[first_dead]
            if stay <= entry:
                new[first_dead], back[t, first_dead] = stay + cost[t, 2], first_dead
            else:
                new[first_dead], back[t, first_dead] = entry + cost[t, 2], from_team
        else:
            new[first_dead], back[t, first_dead] = entry + cost[t, 2], from_team
            # Walk the chain one step, and let the last state hold.
            new[first_dead + 1:last_dead + 1] = total[chain] + cost[t, 2]
            back[t, first_dead + 1:last_dead + 1] = chain
            if total[last_dead] < total[last_dead - 1]:
                new[last_dead] = total[last_dead] + cost[t, 2]
                back[t, last_dead] = last_dead
        total = new

    path = np.empty(n, dtype=np.int16)
    # A half may not end inside an unfinished stoppage chain unless that is all there is.
    ends = total.copy()
    ends[first_dead:last_dead] = np.inf
    path[-1] = int(np.argmin(ends)) if np.isfinite(ends).any() else int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.minimum(path, 2).astype(np.int8)
