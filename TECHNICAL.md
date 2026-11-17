# Possession from tracking: a cost-based decoder with a player-shape fallback

Team: **tacticism** · More Than A Hack 2027, Initial Challenge, BAR-ATH · submitted file: version 4

## 1. Summary

We label each of the 199,319 frames `H`, `A` or `D` with one Viterbi decode over three states:
Barça in possession, Athletic in possession, dead ball. Every frame gives each state a cost, and
every switch between states has a cost, so a label only changes when the evidence for a change
outweighs the price of changing. The evidence comes from four places:

1. **A player on the ball**: who is clearly playing it.
2. **Signs that play has stopped**: players slowing to a walk, the ball outside the lines.
3. **The players' shape**, for the 41% of graded frames where the ball is not tracked.
4. **Restarts found from the ball itself**, and a minimum length for any stoppage.

The submitted file uses the tracking data and the Smart Tagging file only. It does not use the
video. It is rebuilt exactly, byte for byte, by `python v4.py` (section 7).

Scores of each version on the public 5 minutes (17 possessions), from the organisers' scorer:

| version | what it added | macroF1 | spellF1 |
|---|---|---|---|
| organisers' best baseline (`baseline_smooth50`) | nearest player, majority vote | 0.621 | 0.326 |
| v1 | player on the ball, Viterbi decode, a missing ball keeps the team | 0.546 | 0.338 |
| v2 | dead-ball evidence (player speed, ball out, kicks, tags) | 0.812 | 0.490 |
| v3 | player-shape model for the blind stretches | 0.813 | 0.516 |
| **v4 (submitted)** | minimum stoppage length, restarts from the ball | **0.816** | **0.516** |

Five minutes and 17 possessions is too little to rank versions: a bootstrap over the possessions
puts the noise on spellF1 at roughly 0.3. We read these numbers as a sanity check, not as a
leaderboard, and judged each step mainly on whole-match checks (section 5).

## 2. The approach

**Why a decoder and not a per-frame classifier.** A possession is an interval. The baseline's
sliding vote flickers (26 possessions against 17 in the answer key at `smooth50`), and any
per-frame classifier needs hand-made smoothing afterwards. A decode with an explicit price for
each switch builds the rule "keep the last team until the other team clearly has control" into
the method.

**States and costs.** Three states: `H`, `A`, `D`. Each frame adds a cost to the states the
evidence argues against. A switch costs 8 between `H` and `A` and 5 into or out of `D`
(`v1.py`). A player on the ball contradicting the current team costs 1 per frame, so about
8 frames (0.3 s) of the other team clearly on the ball are needed to change team. The decode runs
separately on each graded half.

**Layer 1: who is on the ball** (`v1.py`). A frame is evidence for a team when a player of that
team is within 1.5 m of the ball, no opponent is as close, and the ball moves slower than 7 m/s
(under control rather than in flight). A missing ball is no evidence for either team, so the team
that had the ball keeps it.

**Layer 2: has play stopped** (`v2.py`). Dead-ball evidence is the ball more than 1 m outside the
lines, and the median player speed, smoothed over one second, dropping below 1.4 m/s (the middle
between about 1.1 m/s in the public window's stoppages and about 2 m/s in its play). A ball kicked
faster than 5 m/s is evidence of live play. The Smart Tagging set-piece tags are only a weak nudge
(towards `D` before the tag, towards the tagged team just after): in the public window a restart
falls anywhere from 12 s before to 4 s after the tag start. Before the second-half kickoff anchor
(frame 113741, from `games.py`) the ball is dead.

**Layer 3: the players' shape** (`shape.py`, `v3.py`). On 41% of graded frames the ball is not
tracked, and a change of possession inside such a stretch would be found late or never. A small
gradient-boosted model (300 trees of depth 4, fixed seed) reads 84 numbers describing both teams'
shape without the ball: how far up the pitch each team is, how wide, how compact, how fast it runs
forward. The attacking team spreads out and pushes up, the defending team squeezes and drops back.
It is trained on this match's frames where the ball is visible and layer 1 names a team, so no
hand labelling is involved. It is cross-fitted (a model trained on one half scores the other half),
and the public 5 minutes are left out of training. Its opinion enters the decode as log-odds,
capped at 95/5, at a small weight per frame, so a player on the ball always outvotes it and it
decides only the blind stretches.

**Layer 4: restarts and minimum stoppage** (`decode.py`, `v4.py`). Every stoppage must last at
least 2.5 s inside the decode; version 3 produced many one-second stoppages (a ball briefly tracked
over the line, a lull) that cut real possessions in two. Restart moments are found from the ball:
it returns to the pitch after at least 10 frames outside within 10 s (throw-in, corner, goal kick),
or it sits still for 1.5 s and is then struck faster than 5 m/s (free kick, kick-off). The second
before a restart is pushed towards `D` and the second after towards live play.

## 3. The possession rules we implemented

- Live play always has exactly one team in possession, `H` or `A`. A dead ball is `D`. We do not
  try to say who "owns" a dead ball.
- **Loose and contested balls.** Possession stays with the last team that had it. A frame only
  counts as evidence when exactly one team is close to the ball and the ball is not travelling;
  a duel with both teams close, or a ball in flight, is no evidence, and neither is a missing ball.
- **Challenges and touches.** A single touch or deflection cannot flip the label, because a switch
  needs roughly 8 frames of the other team clearly on the ball, or a sustained opinion from the
  shape model. We do not model passes or "keeps the ball for a few seconds" explicitly; the switch
  price is our proxy for controlled possession.
- **Restarts.** Dead before the restart, live after it, with the restart located from the ball
  (section 2, layer 4). The restarting team is whichever team has a player on the ball first, or
  the shape model's opinion if the ball is not tracked.

## 4. Assets used, tried and dropped

Used: the tracking (players and ball); the Smart Tagging file (weak set-piece nudge, and its
attacking phases as a whole-match sanity check, never as an input to the labels); the published
second-half kickoff anchor; the public ground truth, only to score the public window and to read
off the stoppage and play speeds in layer 2. No possession statistics for this fixture were used.
Public facts about the match (goal times, counts of fouls and corners) served only as coarse
sanity checks, never as inputs.

Tried and dropped:

- **A long gap in the ball track means dead ball** (v1). 41% of graded frames sit in gaps longer
  than 4 s, mostly live play; the rule made 44% of the match dead against about 20% in the public
  window.
- **Cheaper team switches** to catch short possessions. It produced short possessions (30% under
  4 s, like the public 35%) but the wrong ones: the public score and the agreement with the tags
  both fell. In the public window the short possessions mostly have no player of that team close
  to the ball, so the tracking cannot show them.
- **A ball detector on the video (v5, code not included, not submitted).** A YOLO11s detector was
  fine-tuned on frames labelled for free: the tracking's ball position projected through the
  homography and snapped onto a white blob.
  It found a ball on 42% of the frames the tracking lost, and the whole decode changed on 4% of
  frames. The public spellF1 stayed at 0.516 and macroF1 rose by 0.011. We did not submit it:
  detector output differs slightly between machines (a 2-minute test changed 51 of the decode's
  frames), so the file could not be regenerated exactly, and the trained weights are not part of
  the data bundle.

## 5. Validation

The public window can tell us little, so `checks.py` compares each version with the whole match
on quantities we can check without answers. For v4:

| check (alive frames unless noted) | v4 whole match | v4 public window | public answers |
|---|---|---|---|
| agrees with the tagging's attacking phase | 88.4% | 71.3% | 75.4% |
| share of the match that is dead ball | 48.8% | 21.3% | 19.6% |
| possessions per minute of live play | 4.18 | 3.56 | 4.23 |

Caveats: the tagging also reads player positions, so agreement with it may flatter a method that
also reads positions; and the whole-match dead share depends on the speed threshold (1.2 m/s gave
37%, 1.6 m/s gave 57-60%). We expect a professional match to be dead for roughly 45-50% of the
time. The shape model's cross-fit and the exclusion of the public window keep the public score
honest.

## 6. Limitations and what we would do next

- **Short possessions.** 6 of the 17 public possessions are under 4 s and two are under 1.5 s;
  the two shortest happen in crowded duels with both teams' players within 1-2 m of the ball, where
  the tracking shows no clear player on it. Of the 20 real changes in the public window, v4 places
  7 within 2 s, 4 more within 3 to 7 s, and misses 9.
- **The answer key is mechanical** (derived from corrected tracking), so brief flips inside duels
  are not predictable from the rules or the geometry we read.
- **Stoppages.** v4 finds about 150 stoppages against roughly 100-110 by our estimate from the
  match's recorded events: some brief interruptions are probably over-split.
- **Little to validate on.** With more time we would build a larger local check (hand labels are
  allowed to validate a method, not to produce the submission) and tune the switch prices and the
  speed threshold on it.
- **Video.** A ball detector with bit-reproducible inference (fixed precision, CPU or one pinned GPU)
  and published weights could fill the lost-ball frames. The self-labelled training worked; the
  open problem is exact reproduction.

## 7. Reproduction

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git clone https://github.com/Oyewole-Temiloluwa/mtah-2027-possession possession
python3.12 -m venv .venv && .venv/bin/pip install -r possession/requirements.txt
export MTAH_DATA_DIR=/path/to/the/unzipped/data
.venv/bin/python possession/v4.py --out tacticism.csv
.venv/bin/python more-than-a-hack-2027/evaluation/validate_submission.py tacticism.csv
```

We rebuilt the file from a fresh copy of this repository, in a new virtual environment installed
from `requirements.txt`, and compared it with the submitted file using `cmp`:

| setup | result |
|---|---|
| macOS (Apple M1 Pro), Python 3.12.13; the raw tracking parsed from scratch | identical |
| the same, forced to one thread (`OMP_NUM_THREADS=1`) | identical |
| Linux arm64, Docker `python:3.12-slim` | identical |
| Linux x86-64 (emulated), Docker `python:3.12-slim` | identical |

The two Linux runs reused the parsed-tracking cache written by the macOS run, so they test the shape
model and the decoder on other hardware; the parse itself, which is plain text to numbers, was
exercised from scratch on the Mac. End to end, including the parse, the run takes about 1.5 minutes
on an M1 Pro.

SHA-256 of the submitted file, to check a rebuild against:
`d077ad4bce83cf75ebe2be51fb7330c498ea5b4cbd59dcc7dd3320239de4b8ac`
