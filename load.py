"""Read the tactical tracking (ATD) into numpy arrays, and cache them.

The raw text is ~200 MB and takes a while to parse, so the arrays are saved once as a .npz next
to the data (never inside a repository) and reused after that.

    ball     (n_frames, 2)      x, y in metres, NaN where the ball is not seen
    players  (n_frames, 46, 2)  x, y in metres per track slot, NaN where the slot is empty
    team     (46,)              0 = home (Barcelona), 1 = away (Athletic)

Row i of every array is video frame i, i.e. the same index as a submission row.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHALLENGE_REPO = os.environ.get(
    "MTAH_REPO", os.path.join(os.path.dirname(HERE), "more-than-a-hack-2027"))
sys.path.insert(0, CHALLENGE_REPO)
from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

HOME, AWAY = 0, 1


def team_of_each_slot(meta_path):
    """0/1 per track slot. The first team in the metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def parse_atd(raw_path, n_slots, n_frames):
    """Every line is '<frame>:<x,y,s>;...;<x,y,s>:<x,y,s>'. Turn the whole file into one array."""
    with open(raw_path) as handle:
        text = handle.read()
    flat = np.array(text.replace(":", ",").replace(";", ",").split(), dtype=object)
    per_line = 1 + 3 * n_slots + 3
    rows = np.array([np.array(line.split(","), dtype=np.float64) for line in flat])
    if rows.shape != (n_frames, per_line):
        raise ValueError("unexpected ATD shape {}".format(rows.shape))
    if not np.array_equal(rows[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame ids are not 1..{}".format(n_frames))
    players = rows[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = rows[:, 1 + 3 * n_slots:]
    return players, ball


def load(game=GAME, data_dir=None, refresh=False):
    paths = asset_paths(game, data_dir)
    cache = os.path.join(os.path.dirname(paths["atd_raw"]), "..", "cache",
                         "{}_tactical_atd.npz".format(game))
    if os.path.exists(cache) and not refresh:
        stored = np.load(cache)
        return {key: stored[key] for key in stored.files}

    length, width = GAMES[game]["pitch"]
    team = team_of_each_slot(paths["atd_meta"])
    players, ball = parse_atd(paths["atd_raw"], len(team), source_frames("tactical", game))
    scale = np.array([length, width])
    out = {
        "ball": (ball[:, :2] * scale).astype(np.float32),
        "players": (players[:, :, :2] * scale).astype(np.float32),
        "team": team,
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    data = load(refresh="--refresh" in sys.argv)
    seen = np.isfinite(data["ball"][:, 0])
    print("frames {}  slots {} (home {}, away {})  ball seen {:.1%}".format(
        len(data["ball"]), len(data["team"]), int((data["team"] == HOME).sum()),
        int((data["team"] == AWAY).sum()), seen.mean()))
