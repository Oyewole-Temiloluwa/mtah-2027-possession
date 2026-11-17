# Possession labelling for BAR-ATH

Our entry to the Initial Challenge of More Than A Hack 2027. For every frame of the tactical video it
answers one question: does Barça have the ball (`H`), does Athletic (`A`), or is the ball dead (`D`)?

The submitted file is **version 4** (`v4.py`). It is built from the tracking data and the Smart
Tagging file only. It does not use the video, and it contains no learned weights: the one small
model in it (`shape.py`) is trained from scratch on every run. The method is described in
[TECHNICAL.md](TECHNICAL.md).

No match data is in this repository, and none is needed beyond the organisers' data bundle.

## Reproduce the submitted file

You need Python 3.12 and the organisers' data bundle. Put this repository and the
[challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027) side by side (or
set `MTAH_REPO` to wherever the challenge repository is).

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git clone https://github.com/Oyewole-Temiloluwa/mtah-2027-possession possession

python3.12 -m venv .venv
.venv/bin/pip install -r possession/requirements.txt

export MTAH_DATA_DIR=/path/to/the/unzipped/data      # the folder that holds tactical/
.venv/bin/python possession/v4.py --out my_submission.csv

.venv/bin/python more-than-a-hack-2027/evaluation/validate_submission.py my_submission.csv
```

The first run parses the 195 MB tracking file and caches it as an `.npz` next to the data
(`$MTAH_DATA_DIR/cache/`, never inside a repository). Later runs reuse the cache. 

End to end, including the parse, it takes about 1.5 minutes on an M1 Pro. The submitted file's SHA-256 is
`d077ad4bce83cf75ebe2be51fb7330c498ea5b4cbd59dcc7dd3320239de4b8ac`; rebuilds were checked identical on macOS, Linux arm64 and Linux x86-64 (TECHNICAL.md, section 7).

The result is the same file on every run. Nothing is random: the shape model has a fixed seed, and
the decoder is exact dynamic programming.

## What is in here

Used for the submitted file:

| file | what it does |
|---|---|
| `load.py` | reads the tracking (players, ball) into arrays and caches them |
| `v1.py` | who is on the ball, and a Viterbi decode over H / A / D that keeps the last team when unsure |
| `v2.py` | dead-ball evidence: players slowing down, ball outside the lines, kicks, set-piece tags |
| `shape.py`, `v3.py` | a small model that guesses the team from the players' shape when the ball is not tracked |
| `decode.py`, `v4.py` | minimum stoppage length inside the decode, and restart moments found from the ball |
| `checks.py` | whole-match sanity checks (see TECHNICAL.md, section 5) |

Things we tried and dropped, including a ball detector on the video, are described in
[TECHNICAL.md](TECHNICAL.md), section 4. Their code is not part of this repository: nothing above
depends on it.

## Data

The match data is gated and is not redistributable, so it is not here. `.gitignore` keeps data, caches
and CSVs out of the repository.
