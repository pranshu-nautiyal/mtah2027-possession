# Ball possession from tracking data (More Than A Hack 2027)

IMSA More Than a Hack Team: Pranshu Nautiyal, Dhairya Jaswani, Tanush Shastry.

Our entry to the Initial Challenge of [More Than A Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) (Metrica Sports, FC Barcelona Innovation Hub, MWCapital). It labels every frame of BAR-ATH (199,319 frames) as `H` / `A` / `D` / `X` from the tactical tracking data alone. Runs on a CPU in about 20 seconds. Deterministic.

**Result: 125th of 759, spellF1 0.579.** The top 20 advanced, with a cutoff of 0.652. See [RESULTS.md](RESULTS.md) for how our validation compared to the real score and what we'd change.

## Reproduce the submitted CSV

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --data <path to the official data/ folder> --out "IMSA More Than a Hack Team.csv"
```

`--data` is the folder holding `tactical/BAR-ATH_tactical_atd.txt` and `.xml`, as the organisers distributed them. They're not included here because the data isn't redistributable. The script prints the SHA-256 of what it wrote. The submitted file has

```
sha256 bbeb13e4c503bd52075795a7d5967dc0104cef04ee715ef62c005e4c45f46a06
```

Tested with Python 3.14 on macOS (Apple silicon). A fresh clone with a new environment reproduced it byte for byte.

## How it works

1. **Parse** the raw ATD tracking (`possession/atd.py`).
2. **Features per frame** (`possession/features.py`): ball position, speed and visibility, each team's nearest-player distances to the ball, "touch" cues (a player arriving as the ball changes direction or speed), pitch zone. Computed twice, as given and with the teams swapped, so home and away are treated the same.
3. **Stage 1** (LightGBM): a frame classifier (H/A/D) and a change-point detector (switch, stop or restart within ±3 frames).
4. **Stage 2** (LightGBM): refines the frame probabilities from their own context.
5. **Decoder** (`possession/decode.py`): Viterbi over H/A/D. Changing state gets cheaper wherever the change detector fires.

The full write-up we submitted is in [docs/technical_document.md](docs/technical_document.md).

## Training without labels from the match

No BAR-ATH labels were used for anything. We trained and chose every setting on 10 open matches, 7 from IDSSE/DFL (CC BY 4.0) and 3 Metrica samples. Each was degraded by `proxy/dirty.py` to look like BAR-ATH's tracking, using only unlabelled statistics of that tracking. Leave-one-match-out on these gave 0.711. The real score was 0.579, and [RESULTS.md](RESULTS.md) goes into why.

Retrain with `training/train_v3.py <proxy work dir> v6 v3`, with `EXTRA_FEAT_DIRS` pointing at the extra noise seeds. The open datasets aren't redistributed here.

## Layout

```
run.py            the one command that produces the submission
possession/       parser, features, models, decoder
weights/          the submitted v6 models and decoder settings
training/         training scripts
proxy/            proxy matches, noise simulator, experiments
docs/             technical document as submitted
RESULTS.md        official result, validation post-mortem, lessons
```
