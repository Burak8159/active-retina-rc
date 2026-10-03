# MFIDDR Access Log

Date: 2026-10-01

Public source checked: `https://github.com/mfiddr/MFIDDR`

Observed:

- Repository is public and carries an MIT license for repository contents.
- README describes the dataset and file layout.
- Full dataset access is through a linked Google form/path, not present in this local workspace.
- Local search under the current Codex workspace found no `MFIDDR` directory or matching files.

Current A0 state: `BLOCKED`

Required to unblock:

- Obtain the full MFIDDR dataset through the maintainers' access path.
- Place files at `work/active_retina/data/MFIDDR`.
- Keep raw files outside git.
- Re-run `python -m active_retina.prepare_mfiddr --config configs/mfiddr.yaml`.

