# Pluribus hand-history archive

Measured November 12, 2026 with the package's own `download_pluribus` and `replay_pluribus`
(`src/poker_decision/source_experiments.py`). These properties matter when building chronological
Hold'em examples from the archive, such as opponent histories that follow players across hands.

## Contents

| Property | Value |
| --- | --- |
| Archive | PHH v1 from Zenodo, 4,806,967 bytes, SHA-256 pinned as `PHH_SHA256` |
| Hands | 10,000, stored as `data/pluribus/<folder>/<n>.phh` |
| Folders | 92, of which 25 end in `b` |
| Players | Pluribus and 13 human aliases |
| Blinds and stacks | 50/100 chips, and 10,000 chips for every player at the start of every hand |
| Replay | All 10,000 hands replay with `replay_pluribus`, giving 91,356 betting decisions with no failures |

Hands played by each player: Pluribus 10,000; MrBlue 9,121; MrOrange 7,512; Bill 6,713;
MrPink 6,055; Eddie 5,510; MrWhite 4,483; MrBlonde 2,560; Budd 2,509; Joe 1,535; MrBrown 1,378;
Hattori 1,365; ORen 771; Gogo 488.

## Consequences for building examples

- **A folder is one table's session, in order.** Every hand in a folder has the same six players.
  Each file's `hand` field equals its file number, so sorting by file number gives play order.
- **A `b` folder continues its plain folder.** All 25 `b` folders pick up hand numbering exactly
  where the plain folder with the same number stops. For example, `100` holds hands 0–70 and `100b`
  starts at hand 71. Each pair shares five of six players (19 pairs) or three (6 pairs). Treat a
  pair as one session when splitting by session, or the same session can land on both sides of
  the split.
- **Every hole card is in the file.** Each hand lists all six players' hole cards in its `d dh`
  actions, so an input builder must read only the observer's own cards. `pluribus_inputs` already
  reads only the acting player's cards.
- **`fold` is offered when checking is free.** PokerKit's `can_fold()` is true even with nothing to
  call, so `pluribus_inputs` includes `fold` in those spots. That happens in 22,135 of the 91,356
  decisions, and no player folds in any of them. Removing `fold` there would change the shared
  candidate contract, so it needs a team decision. Until then, a model trained on these candidates
  can put probability on a choice that never occurs.
