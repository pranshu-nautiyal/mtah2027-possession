# IMSA More Than a Hack Team — ball possession from tracking data (More than a Hack 2027, Phase 1)

Labels every frame of BAR-ATH (199,319 frames) as `H` / `A` / `D` / `X` from the provided
tactical tracking data only. Inference runs on a CPU in about 20 seconds and is deterministic.

## Reproduce the submitted CSV

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --data <path to the official data/ folder> --out "IMSA More Than a Hack Team.csv"
```

`--data` is the folder that contains `tactical/BAR-ATH_tactical_atd.txt` and
`tactical/BAR-ATH_tactical_atd.xml` (as distributed by the organisers; not included here).
The script prints the SHA-256 of what it wrote. The submitted file has

```
sha256 bbeb13e4c503bd52075795a7d5967dc0104cef04ee715ef62c005e4c45f46a06
```

Tested with Python 3.14 on macOS (Apple silicon). No GPU, no network, no randomness at inference.

## What it does

1. **Parse** the raw ATD tracking (`possession/atd.py`).
2. **Per-frame features** (`possession/features.py`): ball position/speed/visibility, distances
   from the ball to the nearest players of each team, ball-player approach cues ("touches"),
   pitch zone. Computed twice: as given, and with the two teams swapped, so the models are
   symmetric between home and away.
3. **Stage 1** (LightGBM, `weights/v6_frame.txt`, `weights/v6_change.txt`): a frame classifier
   (H / A / D) and a change-point detector (switch / stop / restart within ±3 frames).
4. **Stage 2** (LightGBM, `weights/v6_stack.txt`): refines the frame probabilities from their
   own temporal context.
5. **Decoder** (`possession/decode.py`): Viterbi over H/A/D where the cost of changing state is
   lowered where the change detector fires (`weights/v6_decoder.json`).

## How it was trained (no labels from the provided game)

No BAR-ATH labels of any kind were used for training, tuning or model selection; the public
5-minute ground truth was used only as a format sanity check.

- **Training data:** the 7 open DFL/IDSSE matches (CC BY 4.0; approved by the organisers in
  issue #16) and Metrica Sports' 3 open sample games, converted to the ATD format
  (`proxy/convert_*.py`). IDSSE possession labels were harmonised to event-confirmed
  possession (`proxy/harmonize_idsse.py`).
- **Realistic noise:** those clean matches were degraded to look like the BAR-ATH tracking
  (`proxy/dirty.py`, profile `proxy/noise_profile.json`): smoothing, ball dropouts that depend on
  ball speed, crowding and location, off-pitch episodes, player dropouts. The noise profile is
  fitted only to unlabelled statistics of BAR-ATH's tracking.
- **Model selection:** leave-one-match-out on these 10 matches, scored with the official
  metric, with a paired bootstrap; a change was kept only if it won beyond the noise.
  v6 = three noise seeds per match. Proxy spellF1 0.711 (90% CI 0.695–0.726).
- **Retraining:** `training/train_v3.py <proxy work dir> v6 v3` with
  `EXTRA_FEAT_DIRS` pointing at the extra noise seeds (see the technical document).
  The raw open datasets are not redistributed here.

## Layout

```
run.py            the one command that produces the submission
possession/       parser, features, models, decoder (inference code)
weights/          committed v6 models and decoder settings
training/         training scripts
proxy/            proxy-board construction, noise simulator, experiments
```
