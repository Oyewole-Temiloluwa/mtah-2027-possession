"""Version 1: keep the same team when the ball disappears, and stop the flip-flopping.

Two changes to the reference baseline (nearest player + majority vote):

  1. A frame only counts as evidence when a player is clearly ON the ball: within TOUCH_RADIUS,
     no opponent equally close, and the ball moving slowly enough to be under control. A ball
     flying past a defender, or a missing ball, is no evidence for either team -- so the team
     that had it keeps it.
  2. The labels come from a Viterbi decode over three states (H, A, D) instead of a vote. Every
     switch has a fixed cost, so a switch only happens when the evidence for it outweighs that
     cost. Few flips, and the ones that happen land where the evidence changes.

Dead ball, kept deliberately simple here (proper stoppage detection is version 2):
  - the ball seen outside the pitch lines pushes towards D,
  - the second half is D until its kickoff, a published anchor in games.py.

Tried and dropped: turning long ball gaps into D. 41% of graded frames sit inside a gap longer
than 4 s, most of them in live play, so the rule labelled 44% of the match dead (the public window
has 20%). A missing ball is therefore no evidence at all here -- MISSING_COST is 0.

    python v1.py            writes v1.csv into the challenge repo's evaluation/results/
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load import AWAY, CHALLENGE_REPO, HOME, load  # noqa: E402
from games import GAME, GAMES, anchor_video_frame, half_video_blocks  # noqa: E402

# Params. Chosen from football sense, not tuned on the public five minutes (see EVALUATION.md).
FPS = 25
TOUCH_RADIUS = 1.5       # m: a player this close to the ball can be playing it
CONTROL_SPEED = 7.0      # m/s: faster than this the ball is travelling, not being controlled
OUTSIDE_MARGIN = 1.0     # m beyond the lines before a ball counts as out
SWITCH_TEAM = 8.0        # cost of H <-> A: needs ~8 frames (0.3 s) of the other team on the ball
SWITCH_DEAD = 5.0        # cost of entering or leaving D; > SWITCH_TEAM / 2 so D is never a shortcut
MISSING_COST = 0.0       # per frame with no ball, for H and A: 0 = a missing ball keeps the team
OUTSIDE_COST = 1.0       # per frame of ball outside the lines, for H and A
LIVE_COST = 0.3          # per frame of ball seen on the pitch, for D

H_STATE, A_STATE, D_STATE = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def ball_speed(ball):
    """m/s from a centred difference over +/-2 frames, falling back to +/-1. NaN if unknown."""
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    for step in (2, 1):
        moved = np.linalg.norm(ball[2 * step:] - ball[:-2 * step], axis=1) * FPS / (2 * step)
        fill = np.isnan(speed[step:-step])
        speed[step:-step][fill] = moved[fill]
    return speed


def touch_evidence(ball, players, team):
    """Per frame: HOME or AWAY when exactly one team is on a controlled ball, else -1."""
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    near_home = distance[:, team == HOME].min(axis=1) < TOUCH_RADIUS
    near_away = distance[:, team == AWAY].min(axis=1) < TOUCH_RADIUS

    speed = ball_speed(ball)
    controlled = ~(speed > CONTROL_SPEED)  # unknown speed is given the benefit of the doubt

    out = np.full(len(ball), -1, dtype=np.int8)
    out[near_home & ~near_away & controlled] = HOME
    out[near_away & ~near_home & controlled] = AWAY
    return out


def frame_costs(ball, players, team, game=GAME):
    """(n_frames, 3) cost of each state on each frame. Lower is more consistent with the data."""
    length, width = GAMES[game]["pitch"]
    seen = np.isfinite(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    outside = seen & ((x < -OUTSIDE_MARGIN) | (x > length + OUTSIDE_MARGIN)
                      | (y < -OUTSIDE_MARGIN) | (y > width + OUTSIDE_MARGIN))
    touch = touch_evidence(ball, players, team)

    cost = np.zeros((len(ball), 3), dtype=np.float64)
    cost[touch == AWAY, H_STATE] += 1.0
    cost[touch == HOME, A_STATE] += 1.0
    cost[~seen, H_STATE] += MISSING_COST
    cost[~seen, A_STATE] += MISSING_COST
    cost[outside, H_STATE] += OUTSIDE_COST
    cost[outside, A_STATE] += OUTSIDE_COST
    cost[seen & ~outside, D_STATE] += LIVE_COST

    # Before the second-half kickoff the ball is dead by definition.
    second_half_start = half_video_blocks(game)[1][1]
    kickoff = anchor_video_frame(GAMES[game]["kickoff_2nd"])
    cost[second_half_start:kickoff + 1, :D_STATE] = np.inf
    return cost


def transition_costs():
    switch = np.array([[0.0, SWITCH_TEAM, SWITCH_DEAD],
                       [SWITCH_TEAM, 0.0, SWITCH_DEAD],
                       [SWITCH_DEAD, SWITCH_DEAD, 0.0]])
    return switch


def viterbi(cost, switch):
    """Cheapest state sequence: sum of per-frame costs plus a cost for every switch."""
    n, k = cost.shape
    columns = np.arange(k)
    back = np.empty((n, k), dtype=np.int8)
    total = cost[0].copy()
    for t in range(1, n):
        options = total[:, None] + switch          # options[i, j]: be in i, move to j
        back[t] = np.argmin(options, axis=0)
        total = options[back[t], columns] + cost[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmin(total)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(data, game=GAME):
    """Full-length array of H/A/D labels, X outside the two halves."""
    cost = frame_costs(data["ball"], data["players"], data["team"], game)
    switch = transition_costs()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = LETTERS[viterbi(cost[first:last + 1], switch)]
    return labels


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(CHALLENGE_REPO, "evaluation", "results",
                                                      "v1.csv"))
    args = parser.parse_args()

    labels = predict(load())
    write_submission(args.out, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.out, **counts))
