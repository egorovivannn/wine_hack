# Wine scanner: working instructions

## Goal and deadline

Build a reproducible local service that identifies the correct `slug` from a real photograph of a Russian wine bottle, then shows its catalog card in a mobile interface. Recognition quality is the main competition criterion. The submission cutoff is 29 September 2026, 23:59 Moscow time; confirm any later schedule changes with the organizers before relying on them.

## Data and identities

- `df_2.csv` is a derived catalog table. `slug` identifies a wine card; repeated rows do not create new classes. Do not infer the correct wine from query filenames, file order, or the numeric prefixes of the official photo archive.
- Raw inputs live in the ignored `data/` directory: `data/source/Датасет.zip`, `data/source/official_real_photos.zip`, and `data/field/real_photos.zip`. The latter contains our own store photos and prior detector outputs. Check actual contents before use.
- The official photo archive has no published ground-truth `slug` table. Treat it as unlabeled until identities are verified. Keep manual labels, their evidence, and an explicit unknown/ambiguous state in a separate manifest.
- For the 13 store photos in `data/field/photos`, the user specified the central bottle as the target. `evaluation/field_center_labels.tsv` records the reviewed cases; do not invent an exact catalog `slug` for an absent wine or choose between two bottles at the frame center.
- Verify each `slug -> reference image` join. Strapi filenames differ from the original CSV names and some matches are ambiguous. Never choose the first fuzzy filename match silently.
- Keep raw archives, extracted photos, model weights, caches, and personal photos out of Git. Commit code, small manifests with provenance, and aggregate reports only. Never commit passwords or API keys.

## Evaluation

- The organizer script sends one multipart `image` at a time and reads a flat JSON response with a nonempty `slug`. Keep the evaluation endpoint compatible with the supplied script. Clarify unknown-wine behavior with the organizers before changing its contract.
- Report exact top-1 `slug` accuracy, hit@5, errors on similar vintages/series, and p50/p95 latency on verified labeled real photos. Compute F1 only from a labeled set, not as a per-image confidence value.
- Keep training, threshold tuning, and final evaluation photos independent by capture session. Do not report smoke-test or external-dataset retrieval metrics as hackathon accuracy.
- Record model/checkpoint hash, catalog manifest hash, code revision, hardware, and evaluation population with each result. Preserve error examples for review.

## Implementation and hardware

- Current target host is WSL2 with an RTX 3080 (10 GiB VRAM), Ryzen 5 5600, and about 58 GiB system RAM. Check available memory before each training plan. Earlier 24 GiB GPU results in this repository do not establish that the same settings fit here.
- Favor an end-to-end runnable baseline first: decode/orient photo, identify target bottle/label, retrieve reference candidates, distinguish close cards, return one `slug`, and display its card. Add model training only after measuring a clear baseline and reviewing failure cases.
- Keep preprocessing and reference mapping deterministic. Make paths configurable; do not rely on prior developers' absolute paths or missing checkpoints.
- Run relevant tests and the organizer's evaluation script after changes to inference. Document exact setup and limitations in `README.md` and architecture decisions in `ARCHITECTURE.md`.

## Collaboration

- Preserve existing work on `main`; develop on a separate branch. Coordinate edits to shared code and evaluation manifests with teammates.
- Before submission, freeze a reproducible commit and verify that all links, setup steps, demo, and required presentation files work from that commit. Do not alter the submitted revision after the cutoff.
