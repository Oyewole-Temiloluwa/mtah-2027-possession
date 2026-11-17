"""Which team has the ball, judged from the players alone -- for when the ball cannot be seen.

The team in possession spreads out and pushes up; the team without it squeezes together and runs
back towards its own goal. This module turns the 22 player positions of every frame into a few
dozen numbers that describe that (shape_features), and trains a small model to read them
(train_shape_model).

The model never sees the ball, so it works the same on frames where the ball is missing. It is
taught from the frames where we are already confident -- the ball is visible and the version 2
decode names a team -- so nobody labels anything by hand. The public five minutes are left out of
the training, so they stay a fair test.
"""

import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load import AWAY, HOME  # noqa: E402
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# Params
FPS = 25
VELOCITY_STEP = 5         # frames: per-player velocity from 0.2 s displacements
SMOOTHING = (25, 75)      # frames: every feature is also given as a 1 s and a 3 s rolling mean
RECOVERING_SPEED = 2.0    # m/s towards their own goal: a player running back
TRAIN_EVERY = 5           # use every 5th confident frame; neighbours are near-duplicates anyway
PUBLIC_WINDOW = (11185, 18685)  # half-open; kept out of training

# A deliberately small, regularised model: a few hundred trees of depth 4 cannot memorise a match.
MODEL_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.05, l2_regularization=1.0,
                    random_state=0)


def attacks_left_to_right(game=GAME):
    """{team code: True if it attacks towards x = length in the first half}, from the metadata."""
    root = ET.parse(asset_paths(game)["atd_meta"]).getroot()
    out = {}
    for code, team in zip((HOME, AWAY), root.iter("Team")):
        for parameter in team.iter("ProviderParameter"):
            if parameter.findtext("Name") == "attack_direction_first_half":
                out[code] = parameter.findtext("Value") == "left_to_right"
    return out


def _team_features(x_attack, y, velocity_attack, speed, length):
    """Per-frame numbers for one team. x_attack runs towards the goal that team attacks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where nobody is tracked
        ordered = np.sort(x_attack, axis=1)  # NaN sorts last
        count = np.isfinite(x_attack).sum(axis=1)
        last_defender = ordered[:, 1]       # [0] is normally the goalkeeper
        most_advanced = np.nanmax(x_attack, axis=1)
        return {
            "count": count.astype(np.float32),
            "mean_x": np.nanmean(x_attack, axis=1),
            "std_x": np.nanstd(x_attack, axis=1),
            "std_y": np.nanstd(y, axis=1),
            "width": np.nanmax(y, axis=1) - np.nanmin(y, axis=1),
            "most_advanced": most_advanced,
            "last_defender": last_defender,
            "depth": most_advanced - last_defender,
            "in_opponent_half": np.nansum(x_attack > length / 2, axis=1).astype(np.float32),
            "mean_speed": np.nanmean(speed, axis=1),
            "forward_velocity": np.nanmean(velocity_attack, axis=1),
            "recovering": np.nansum(velocity_attack < -RECOVERING_SPEED, axis=1).astype(
                np.float32),
        }


def shape_features(data, game=GAME):
    """(n_frames, n_features) DataFrame describing both teams' shape. No ball anywhere in it."""
    players, team = data["players"], data["team"]
    length = GAMES[game]["pitch"][0]
    n = len(players)

    # +1 where a team attacks towards x = length on that frame, -1 the other way. Only the two
    # halves matter; outside them the first-half direction is used and the frames are ignored.
    second_half = np.zeros(n, dtype=bool)
    _, first, last = half_video_blocks(game)[1]
    second_half[first:last + 1] = True
    left_to_right = attacks_left_to_right(game)

    moved = np.full(players.shape, np.nan, dtype=np.float32)
    moved[VELOCITY_STEP:] = (players[VELOCITY_STEP:] - players[:-VELOCITY_STEP]) * FPS / VELOCITY_STEP
    speed = np.linalg.norm(moved, axis=2)

    columns = {}
    for code, name in ((HOME, "home"), (AWAY, "away")):
        sign = np.where(second_half, -1.0, 1.0) * (1.0 if left_to_right[code] else -1.0)
        mask = team == code
        x = players[:, mask, 0]
        x_attack = np.where(sign[:, None] > 0, x, length - x)
        velocity_attack = moved[:, mask, 0] * sign[:, None]
        for key, values in _team_features(x_attack, players[:, mask, 1], velocity_attack,
                                          speed[:, mask], length).items():
            columns["{}_{}".format(name, key)] = values

    raw = pd.DataFrame(columns)
    # The two teams side by side: who is further up the pitch, who is more spread out.
    raw["diff_mean_x"] = raw["home_mean_x"] - raw["away_mean_x"]
    raw["diff_width"] = raw["home_width"] - raw["away_width"]
    raw["diff_depth"] = raw["home_depth"] - raw["away_depth"]
    raw["diff_forward_velocity"] = raw["home_forward_velocity"] - raw["away_forward_velocity"]

    frames = [raw]
    for window in SMOOTHING:
        smoothed = raw.rolling(window, center=True, min_periods=1).mean()
        smoothed.columns = ["{}_{}".format(c, window) for c in raw.columns]
        frames.append(smoothed)
    return pd.concat(frames, axis=1).astype(np.float32)


def training_frames(labels, ball, game=GAME):
    """Frames to learn from: ball visible, the decode names a team, outside the public window."""
    usable = np.isin(labels, ("H", "A")) & np.isfinite(ball[:, 0])
    usable[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = False
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    index = np.flatnonzero(usable & graded)
    return index[::TRAIN_EVERY]


def train_shape_model(features, labels, index):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features.values[index], (labels[index] == "H").astype(int))
    return model


def home_probability(model, features):
    """P(home has the ball) per frame, from the players' shape alone."""
    return model.predict_proba(features.values)[:, 1]
