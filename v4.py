"""Version 4: version 3 with sharper stoppage edges and no fake stoppages.

What changed, and why:

  1. Every stoppage must last at least MIN_DEAD seconds (decode.py). Version 3 produced 38
     stoppages under 2.5 s across the match -- a ball briefly tracked over the line, a lull in
     play -- and each one split a real possession in two.
  2. Restarts are found from the ball itself, so the decode can switch from D to play on the
     right frame instead of a second later, when the players get going again (measured: the
     players' speed only recovers about 1 s after the ball is back in play).
       - the ball comes back onto the pitch after being outside: a throw-in, corner or goal kick
       - the ball sits still for a while and is then struck: a free kick, goal kick or kick-off
     Just before a restart the ball is dead; just after it, it is alive.

Tried and dropped, to catch more short possessions: lowering the cost of a team switch. It did
produce short possessions (30% under 4 s, like the public truth's 35%) but the wrong ones -- the
public score and the tagging agreement both fell. In the public window the short possessions
mostly have no player of that team on a visible ball at all, so the tracking cannot show them.

    python v4.py         writes v4.csv into the challenge repo's evaluation/results/
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
from load import CHALLENGE_REPO, load  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

H_STATE, A_STATE, D_STATE = v1.H_STATE, v1.A_STATE, v1.D_STATE

# Params. Picked from what a restart looks like, not tuned on the public window.
FPS = 25
MIN_DEAD = int(2.5 * FPS)     # frames: the shortest stoppage the decode may produce
OUT_RUN = 10                  # frames outside the lines before it counts as the ball going out
BACK_WITHIN = 10 * FPS        # frames: the ball must come back within this to be a restart
STILL_SPEED = 0.5             # m/s: a ball this slow is sitting still
STILL_RUN = int(1.5 * FPS)    # frames it must sit still before a strike counts as a restart
STRIKE_WITHIN = int(0.5 * FPS)  # frames after the still run in which the strike must come
BEFORE = 1 * FPS              # frames before a restart pushed towards D
AFTER = 1 * FPS               # frames after a restart pushed towards play
RESTART_COST = 1.0            # per frame on each side: as strong as a player on the ball


def restart_frames(ball, game=GAME):
    """Frames where play restarts, found from the ball track alone."""
    length, width = GAMES[game]["pitch"]
    seen = np.isfinite(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    outside = seen & ((x < -v1.OUTSIDE_MARGIN) | (x > length + v1.OUTSIDE_MARGIN)
                      | (y < -v1.OUTSIDE_MARGIN) | (y > width + v1.OUTSIDE_MARGIN))
    inside = seen & ~outside
    out = []

    # 1. Back onto the pitch after a spell outside the lines.
    edges = np.diff(np.concatenate(([0], outside.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < OUT_RUN:
            continue
        back = np.flatnonzero(inside[stop:stop + BACK_WITHIN])
        if len(back):
            out.append(stop + int(back[0]))

    # 2. Struck after sitting still.
    speed = v1.ball_speed(ball)
    still = inside & (speed < STILL_SPEED)
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < STILL_RUN:
            continue
        struck = np.flatnonzero(speed[stop:stop + STRIKE_WITHIN] > v2.KICK_SPEED)
        if len(struck):
            out.append(stop + int(struck[0]))
    return np.array(sorted(set(out)), dtype=np.int64)


def restart_costs(n_frames, restarts):
    cost = np.zeros((n_frames, 3))
    for frame in restarts:
        cost[max(0, frame - BEFORE):frame, :D_STATE] += RESTART_COST
        cost[frame:frame + AFTER, D_STATE] += RESTART_COST
    return cost


def decode_halves(cost, game=GAME):
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        codes = decode.viterbi_min_dead(cost[first:last + 1], v1.SWITCH_TEAM, v1.SWITCH_DEAD,
                                        MIN_DEAD)
        labels[first:last + 1] = v1.LETTERS[codes]
    return labels


def predict(data, game=GAME):
    first_pass = v2.predict(data, game)
    probability = v3.cross_fitted_home_probability(data, first_pass, game)
    cost = v2.frame_costs(data, game) + v3.shape_costs(probability)
    cost += restart_costs(len(cost), restart_frames(data["ball"], game))
    return decode_halves(cost, game)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(CHALLENGE_REPO, "evaluation", "results",
                                                      "v4.csv"))
    args = parser.parse_args()

    labels = predict(load())
    v1.write_submission(args.out, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.out, **counts))
