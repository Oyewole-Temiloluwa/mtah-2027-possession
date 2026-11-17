"""Full-match sanity checks that do not need the private ground truth.

The public five minutes hold 17 possessions, too few to tell two versions apart. These look at the
whole match instead, using things we are allowed to know:

  team agree  Smart Tagging attacking phases (Build Up, Progression, Creating Chances, Attacking
              Transition) name the team with the ball. Of those frames that we call H or A, the
              share where we name the same team. Noisy -- the tags are automatic -- but it covers
              the whole match.
  D in attack share of attacking-phase frames we call dead. Attacking phases are mostly live play,
              so a high value means stoppages invented in the middle of play.
  dead share  share of graded frames labelled D.
  poss/live   possessions per minute of live (non-D) play.

Every column is printed for the whole match and for the public window, and the public ground
truth's own values are printed alongside, so a version can be compared with the truth on the
window and with itself across the match.

    python checks.py results/a.csv results/b.csv ...
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load import CHALLENGE_REPO  # noqa: E402
sys.path.insert(0, os.path.join(CHALLENGE_REPO, "starter"))
sys.path.insert(0, os.path.join(CHALLENGE_REPO, "evaluation"))
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import half_video_blocks  # noqa: E402

ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"}
TEAM_LETTER = {"barcelona": "H", "atletic club": "A"}
PUBLIC_GT = os.path.join(CHALLENGE_REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25


def attacking_frames(n_frames):
    """Array of H/A/'' per frame from the attacking-phase tags."""
    out = np.full(n_frames, "", dtype="<U1")
    for event in load_smart_tagging.load():
        letter = TEAM_LETTER.get(str(event["tags"].get("Team", "")).strip().lower())
        if event["code"].strip().upper() in ATTACKING and letter:
            out[event["start_frame"]:event["end_frame"]] = letter
    return out


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        mask[first:last + 1] = True
    return mask


def read_labels(path):
    return pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values


def report(labels, tagged, mask):
    alive = np.isin(labels, ("H", "A"))
    on_tag = mask & (tagged != "")
    team_agree = (labels[on_tag & alive] == tagged[on_tag & alive]).mean()
    dead_in_attack = (labels[on_tag] == "D").mean()
    dead = (labels[mask] == "D").mean()

    codes = pm.label_codes(labels)
    runs = pm.valid_runs(np.where(mask, 0, pm.X).astype(np.int8))
    spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    live_minutes = (mask & alive).sum() / FPS / 60.0
    return team_agree, dead_in_attack, dead, len(spells) / max(live_minutes, 1e-9), len(spells)


if __name__ == "__main__":
    gt = read_labels(PUBLIC_GT)
    n_frames = len(gt)
    tagged, graded = attacking_frames(n_frames), graded_mask(n_frames)
    window = gt != "X"

    row = "{:<20} {:>10} {:>11} {:>10} {:>9} {:>5}   {:>10} {:>11} {:>10} {:>9} {:>4}"
    print(row.format("", "WHOLE", "MATCH", "", "", "", "PUBLIC", "WINDOW", "", "", ""))
    print(row.format("submission", "team agree", "D in attack", "dead share", "poss/live", "poss",
                     "team agree", "D in attack", "dead share", "poss/live", "poss"))
    fmt = "{:<20} {:>10.1%} {:>11.1%} {:>10.1%} {:>9.2f} {:>5d}   {:>10.1%} {:>11.1%} {:>10.1%} {:>9.2f} {:>4d}"
    print("{:<20} {:>48}   ".format("public truth", "") + "{:>10.1%} {:>11.1%} {:>10.1%} {:>9.2f} {:>4d}".format(
        *report(gt, tagged, window)))
    for path in sys.argv[1:]:
        labels = read_labels(path)
        print(fmt.format(os.path.splitext(os.path.basename(path))[0],
                         *report(labels, tagged, graded), *report(labels, tagged, window)))
