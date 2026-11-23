# Ball possession from tracking data

*Technical document as submitted on 14 Nov 2026 (More Than A Hack 2027, Initial Challenge). The official result, which came later, is in [../RESULTS.md](../RESULTS.md).*

**Team:** IMSA More Than a Hack Team: Pranshu Nautiyal, Dhairya Jaswani, Tanush Shastry\
**Code:** <https://github.com/pranshu-nautiyal/mtah2027-possession>\
**Submitted file:** `IMSA More Than a Hack Team.csv`, SHA-256 `bbeb13e4c503bd52075795a7d5967dc0104cef04ee715ef62c005e4c45f46a06`

## 1. Approach

We label each frame H / A / D from the tactical tracking alone, using a learned pipeline:

1. **Features per frame** (`possession/features.py`): ball position, speed and visibility; for each team, the distance from the ball to its nearest players; "touch" cues (a player arriving at the ball while it changes direction or speed); and pitch zone. Every feature is computed twice, once as given and once with the teams swapped. The model sees both views, so home and away are treated symmetrically.
2. **Stage 1** (LightGBM): a frame classifier (H/A/D) and a change-point detector (does a switch, stop or restart happen within ±3 frames?).
3. **Stage 2** (LightGBM): refines the frame probabilities using their own temporal context.
4. **Decoder**: a Viterbi pass over H/A/D. The cost of changing state drops wherever the change detector fires (cost = base + α·(−log p_change)).

**Why this design.** The ranking metric (spellF1) punishes short wrong-team blips and missed short possessions much more than slightly wrong boundaries. Classifying each frame on its own makes blips, and simple smoothing erases real short possessions. A change-aware decoder handles both.

## 2. Assets used, and assets dropped

| asset | used? | how |
|---|---|---|
| tactical tracking (`BAR-ATH_tactical_atd.txt/.xml`) | **yes** | the only input at inference |
| unlabelled statistics of that tracking | **yes** | to make the training data's noise look like it (§3) |
| tactical / broadcast video | no | never used as input or for checking |
| Smart Tagging | no | not used; we kept the pipeline tracking-only |
| public 5-minute ground truth | format check only | never used to train, tune or choose anything |

## 3. Training data (none from the provided game)

| dataset | source / licence | labelled by | amount |
|---|---|---|---|
| IDSSE (DFL open tracking + events) | Bassek et al., CC BY 4.0, via kloppy | DFL / Sportec event and ball-status data | 7 Bundesliga matches |
| Metrica Sports sample games | Metrica open data (GitHub) | Metrica event data | 3 matches |

Both were converted to the ATD format. Possession labels were derived from each dataset's own events (§4).

**Making clean data look like ours.** The open datasets have much cleaner tracking than BAR-ATH. We wrote a noise simulator (`proxy/dirty.py`) that smooths positions, drops the ball depending on its speed, location and how crowded it is, hides off-pitch episodes, and drops players. Its settings were fitted only to **unlabelled** statistics of the BAR-ATH tracking, such as how often the ball is missing and how fast it accelerates. No labels were involved.

**Model selection.** We left out one of the 10 matches at a time and scored it with the official metric. A change was kept only if a paired bootstrap showed it beating the previous version beyond the noise. As a sanity check, the official baselines scored on these matches rank in the same order as their published scores on BAR-ATH.

| version | change | held-out spellF1 |
|---|---|---|
| baseline (window 50) | none | 0.283 |
| v1 | frame classifier + Viterbi | 0.664 |
| v2 | touch cues + change detector | 0.675 |
| v3 | second-stage model | 0.705 |
| v5 | no player-count features, final noise model | 0.704 |
| **v6 (submitted)** | **3 noise versions of each match** | **0.711 (90% CI 0.695 to 0.726)** |

## 4. Possession rules implemented

- **Who has the ball:** a team gains possession only when one of its players makes an on-ball action (pass, touch, shot, goalkeeper claim, set piece, or a won duel).
- **Loose balls and challenges:** possession stays with the last team that had it until the other team makes such an action. A contested ball that nobody controls does not change possession.
- **Restarts:** while the ball is dead the frame is D. A restart (throw-in, free kick, goal kick, corner, kick-off) gives possession to the team taking it, with nothing carried over from before.
- **Timing:** event times were aligned to the tracking per match, using the moments play restarts.

## 5. Limitations

- **Simulated noise:** the noise is simulated, so the true score on BAR-ATH is unknown. The public 5 minutes are too short to tell.
- **Change detector:** it fires about half as often on BAR-ATH as on the training data. Part of this is genuine, because Barcelona keep the ball, but some real switches are probably missed.
- **Fast passing:** very short possessions under 1.5 s, such as one-touch passing, are still mostly missed.
- **With more time:** we would train the change detector on more varied matches and add a model that identifies the player in possession.

## 6. Reproduction

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --data <official data folder> --out "IMSA More Than a Hack Team.csv"
```

The run takes about 20 s on a CPU, with no GPU, network or randomness. The script prints the output's SHA-256, which should match the one above. We checked this on a fresh clone of the public repository with a new environment. The trained models are in `weights/`. The training and validation scripts are in `training/` and `proxy/`.
