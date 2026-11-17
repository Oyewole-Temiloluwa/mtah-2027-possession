"""Version 3: version 2 plus the players' shape, for when the ball cannot be seen.

Version 2 only learns who has the ball from a player on the ball, so a change of possession
inside a blind stretch (the ball is missing on 4 frames in 10) is found late or not at all.
Version 3 adds a second, weaker witness that never needs the ball: a small model reading the 22
players' shape (shape.py) -- who is spread out and pushing up, who is compact and running back.

How it is wired:
  1. Run version 2. Where the ball is visible and it names a team, that is a training example.
  2. Cross-fit the shape model: train on one half, predict the other, and swap. Every frame's
     shape opinion therefore comes from a model that never trained on that half -- including the
     public five minutes, which are left out of training on top of that.
  3. Add the shape opinion to version 2's frame costs, as log-odds, at SHAPE_WEIGHT per frame.
     A player on the ball still costs 1.0 a frame, so it outvotes the shape whenever there is
     one; the shape decides the blind stretches, and needs a couple of seconds of a clear
     opinion to switch the team on its own.

    python v3.py         writes v3.csv into the challenge repo's evaluation/results/
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shape  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
from load import CHALLENGE_REPO, load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

H_STATE, A_STATE = v1.H_STATE, v1.A_STATE

# Params. Picked before scoring, not tuned on the public window.
SHAPE_WEIGHT = 0.05      # per frame per unit of log-odds: a 0.95 opinion (log-odds 3) costs 0.15
LOG_ODDS_CAP = 3.0       # no single frame says more than 95/5, however sure the model is
PROBABILITY_FLOOR = 1e-4


def cross_fitted_home_probability(data, labels, game=GAME):
    """P(home has the ball) per frame from the shape model, never predicted by a model trained
    on the same half."""
    features = shape.shape_features(data, game)
    index = shape.training_frames(labels, data["ball"], game)
    halves = half_video_blocks(game)
    probability = np.full(len(labels), 0.5)
    for _, first, last in halves:
        held_out = (index >= first) & (index <= last)
        model = shape.train_shape_model(features, labels, index[~held_out])
        rows = slice(first, last + 1)
        probability[rows] = model.predict_proba(features.values[rows])[:, 1]
    return probability


def shape_costs(probability):
    p = np.clip(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    log_odds = np.clip(np.log(p / (1.0 - p)), -LOG_ODDS_CAP, LOG_ODDS_CAP)
    cost = np.zeros((len(p), 3))
    cost[:, H_STATE] = SHAPE_WEIGHT * np.maximum(-log_odds, 0.0)   # shape says away
    cost[:, A_STATE] = SHAPE_WEIGHT * np.maximum(log_odds, 0.0)    # shape says home
    return cost


def predict(data, game=GAME):
    first_pass = v2.predict(data, game)
    probability = cross_fitted_home_probability(data, first_pass, game)
    cost = v2.frame_costs(data, game) + shape_costs(probability)
    switch = v1.transition_costs()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = v1.LETTERS[v1.viterbi(cost[first:last + 1], switch)]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(CHALLENGE_REPO, "evaluation", "results",
                                                      "v3.csv"))
    args = parser.parse_args()

    labels = predict(load())
    v1.write_submission(args.out, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(args.out, **counts))
