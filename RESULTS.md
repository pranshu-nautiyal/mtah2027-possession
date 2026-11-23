# Results and lessons

## Official result

Initial Challenge scoreboard, published by the organisers on 20 Nov 2026:

| | spellF1 |
|---|---|
| **IMSA More Than a Hack Team (this repo, v6)** | **0.579, rank 125 of 759** |
| 1st place | 0.710 |
| 20th place (last spot into the next phase) | 0.652 |
| official baseline (smooth50) | 0.243 on the full match |

Top 17% of the field, and about 2.4x the baseline. But not enough for the top 20, which needed another 0.073.

## How our own validation held up

We never saw a single BAR-ATH label while building this. Everything was chosen on a "proxy board": 10 open matches (7 IDSSE, 3 Metrica) degraded to look like BAR-ATH's tracking. Here's how its predictions compared to the real thing.

| what we measured before submitting | prediction | real |
|---|---|---|
| proxy board, held-out spellF1 | 0.711 | 0.579 |
| public 5-minute window (sanity check only) | 0.581 | 0.579 |
| ranking vs 7 public competitor pipelines | we beat all 7 | all 7 finished below us |

The proxy board ranked methods correctly. We ran 7 other teams' public code on it, and every one finished below us on the real scoreboard, with the board's ordering mostly preserved (best of them: 150th, 0.565). The absolute level was off by 0.13, though.

The 5-minute public window nailed it: 0.581 predicted, 0.579 real. We treated it as too short to trust, because the organisers' scorer gave it a noise band of about ±0.35. In hindsight it carried more signal than that. It also scored v5 (0.619) above v6 (0.581). We shipped v6 because the proxy board preferred it, so v5 may have done slightly better. Still nowhere near the 0.07 gap to the top 20.

## What we think went wrong

1. **The simulated noise was too easy.** Our degradation simulator copied BAR-ATH's ball dropout rates, smoothing and off-pitch gaps. But clearly not everything: a 0.13 drop means the real tracking breaks in ways the simulator didn't reproduce. A domain classifier could still tell BAR-ATH frames from proxy frames almost perfectly (AUC 0.998 on the second board), and we accepted that too easily.
2. **The change detector under-fires on the real match.** It fired about half as often on BAR-ATH as on the proxies. We put that down to Barcelona keeping the ball. Some of it probably was missed switches.
3. **Short possessions.** Possessions under 1.5 s (one-touch passing, quick interceptions) were mostly missed even on the proxies, and spellF1 counts every one of them.
4. **We trusted one validation signal.** When the proxy and the public window disagreed (v5 vs v6), we went with the proxy because it had more data. Better: treat any disagreement as a warning that the proxy is miscalibrated.

## What we'd do next time

- Calibrate the noise simulator until a domain classifier can't separate proxy from real frames, not just until summary statistics match.
- Use the public window as a second, weaker vote in model selection (allowed as a sanity check), and investigate whenever it disagrees with the proxy.
- Model the player in possession, not only the team, to catch short possessions.
- Use the video only as input, which the rules allow, for ball visibility when tracking drops it.

## Timeline

- Nov 3: challenge opens; proxy board and v1 (0.664 on proxy)
- Nov 3 to 11: v2 to v6 (0.675, 0.705, 0.710, 0.704, 0.711 on proxy); competitor benchmark
- Nov 14: submitted v6
- Nov 20: results, 125th of 759
