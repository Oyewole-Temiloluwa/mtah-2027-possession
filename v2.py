"""Version 2: version 1 plus real stoppage detection.

Version 1 knew who had the ball but almost never said the ball was dead (6% of the match, against
about 20% in the public window), so possessions either side of a stoppage were glued together.
Version 2 adds three kinds of dead-ball evidence to the same Viterbi decode:

  1. Players slow down. When the referee stops play, the whole pitch walks. The median player
     speed, smoothed over a second, is about 1.1 m/s in the public stoppages and about 2 m/s in
     play. Below SPEED_MID pushes towards D, above it pushes away, more strongly the further off.
  2. The ball is outside the lines (as in version 1).
  3. Smart Tagging set pieces. A tag lands near a restart, but only roughly -- in the public window
     the restart is anywhere from 12 s before to 4 s after the tag start -- so it is a weak nudge
     towards D before the tag, and towards the tagged team (when there is one) just after.

A ball being kicked hard is evidence of live play, which pulls the restart to the kick rather than
to the moment players get going again. A ball merely seen on the pitch is not: it sits there
during every free kick, which is why version 1's LIVE_COST is gone.

    python v2.py                     writes v2.csv into the challenge repo's evaluation/results/
    python v2.py --no-tags           the same without Smart Tagging, to measure what it adds
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v1  # noqa: E402
from load import AWAY, CHALLENGE_REPO, HOME, load  # noqa: E402
from games import GAME, GAMES, anchor_video_frame, half_video_blocks  # noqa: E402

sys.path.insert(0, os.path.join(CHALLENGE_REPO, "starter"))
import load_smart_tagging  # noqa: E402

H_STATE, A_STATE, D_STATE = v1.H_STATE, v1.A_STATE, v1.D_STATE

# Params. Picked from the measured gap between stoppages and play, not tuned on the public score.
FPS = 25
SPEED_STEP = 5           # frames: player speed from 0.2 s displacements, which smooths jitter
SPEED_SMOOTH = 25        # frames: then averaged over 1 s
SPEED_MID = 1.4          # m/s: halfway between stoppages (~1.1) and play (~2.0)
SPEED_SCALE = 0.4        # m/s: how far from SPEED_MID counts as full-strength evidence
SPEED_WEIGHT = 0.15      # per frame at full strength: a 5 s stoppage at 1.0 m/s outweighs 2 switches
SPEED_CAP = 1.5          # no single frame counts for more than 1.5x full strength
KICK_SPEED = 5.0         # m/s: a ball this fast has been kicked
KICK_COST = 0.2          # per frame of kicked ball, for D
TAG_BEFORE = 10 * FPS    # frames before a set-piece tag where the ball is probably dead
TAG_DEAD_COST = 0.03     # per frame there, for H and A -- a nudge, not a decision
TAG_AFTER = 5 * FPS      # frames after a set-piece tag where the tagged team restarts
TAG_TEAM_COST = 0.1      # per frame there, for the other team

TEAM_STATE = {"barcelona": H_STATE, "atletic club": A_STATE}


def player_speed(players):
    """Median speed of the tracked players per frame, in m/s, smoothed over SPEED_SMOOTH frames."""
    moved = np.full(players.shape[:2], np.nan, dtype=np.float32)
    moved[SPEED_STEP:] = np.linalg.norm(players[SPEED_STEP:] - players[:-SPEED_STEP],
                                        axis=2) * FPS / SPEED_STEP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        median = np.nanmedian(moved, axis=1)
    known = np.isfinite(median)
    window = np.ones(SPEED_SMOOTH)
    total = np.convolve(np.where(known, median, 0.0), window, mode="same")
    count = np.convolve(known.astype(float), window, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def set_piece_costs(n_frames):
    """(n_frames, 3) extra cost from the Smart Tagging set pieces."""
    cost = np.zeros((n_frames, 3))
    for event in load_smart_tagging.load():
        if event["code"].strip().upper() != "SET PIECES":
            continue
        start = event["start_frame"]
        cost[max(0, start - TAG_BEFORE):start, :D_STATE] += TAG_DEAD_COST
        team = TEAM_STATE.get(str(event["tags"].get("Team", "")).strip().lower())
        if team is not None:
            other = A_STATE if team == H_STATE else H_STATE
            cost[start:start + TAG_AFTER, other] += TAG_TEAM_COST
    return cost


def frame_costs(data, game=GAME, use_tags=True):
    ball, players, team = data["ball"], data["players"], data["team"]
    length, width = GAMES[game]["pitch"]
    seen = np.isfinite(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    outside = seen & ((x < -v1.OUTSIDE_MARGIN) | (x > length + v1.OUTSIDE_MARGIN)
                      | (y < -v1.OUTSIDE_MARGIN) | (y > width + v1.OUTSIDE_MARGIN))
    touch = v1.touch_evidence(ball, players, team)
    kicked = v1.ball_speed(ball) > KICK_SPEED

    cost = np.zeros((len(ball), 3))
    # Who has it -- unchanged from version 1. A missing ball is no evidence either way.
    cost[touch == AWAY, H_STATE] += 1.0
    cost[touch == HOME, A_STATE] += 1.0

    # Is it dead?
    cost[outside, :D_STATE] += v1.OUTSIDE_COST
    cost[kicked & ~outside, D_STATE] += KICK_COST
    pace = (player_speed(players) - SPEED_MID) / SPEED_SCALE
    pace = np.clip(np.nan_to_num(pace, nan=0.0), -SPEED_CAP, SPEED_CAP) * SPEED_WEIGHT
    cost[:, :D_STATE] += np.maximum(-pace, 0.0)[:, None]   # slow -> costs H and A
    cost[:, D_STATE] += np.maximum(pace, 0.0)               # fast -> costs D
    if use_tags:
        cost += set_piece_costs(len(ball))

    second_half_start = half_video_blocks(game)[1][1]
    kickoff = anchor_video_frame(GAMES[game]["kickoff_2nd"])
    cost[second_half_start:kickoff + 1, :D_STATE] = np.inf
    return cost


def predict(data, game=GAME, use_tags=True):
    cost = frame_costs(data, game, use_tags)
    switch = v1.transition_costs()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = v1.LETTERS[v1.viterbi(cost[first:last + 1], switch)]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--out", help="output CSV (default: evaluation/results/v2.csv)")
    args = parser.parse_args()

    name = "v2_no_tags.csv" if args.no_tags else "v2.csv"
    out = args.out or os.path.join(CHALLENGE_REPO, "evaluation", "results", name)
    labels = predict(load(), use_tags=not args.no_tags)
    v1.write_submission(out, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(out, **counts))
