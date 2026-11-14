# Source reader correction before fitting

Dated 14 November 2026, after archive checksum verification and inspection of recording headers and annotation tables, but before signal feature inspection, model fitting, policy selection or evaluation of outcomes. The original protocol remains unchanged and is preserved at commit `2879a01f88a9b7d59275491f6a6d48cda2826cc5`.

## Observed source discrepancies

The original Figshare archive has exactly the expected 24 CNT files and the expected 14 EEG channels plus VEOU and VEOL. However, its legacy sample count headers are unreliable. Automatic data format detection in MNE 1.13.2 rejects the first file because its header sample count is negative. Two positive but invalid header counts also imply recordings lasting more than 99 days, despite much shorter file byte extents.

The source annotation tables also differ from the paper's description of 25 trials per class in every run. S1 session 1 run A has 30 of each class; its run B has 30 left, 29 right and 30 feet cues. S2 session 2 run B and S4 session 1 run A have 20 of each class. S3 session 2 run B has 26 feet cues. The other recordings have 25 of each class. These are source annotation counts rather than model results.

## Correction

Use the public MNE reader arguments `data_format="int16"` and `recompute_n_samples=True`. The latter obtains the number of samples from the signal byte extent and the channel count rather than the unreliable sample count field. This leaves the original files and their checksums unchanged. The 16 bit interpretation gives the documented ten second cycle, with median consecutive cue spacing near 9.992 seconds; a 32 bit interpretation would halve this spacing and contradict the five second task plus rest and preparation procedure.

Store exact file sample counts and all source annotation counts in [source_layout.json](source_layout.json), and verify them on every load. Keep all recorded cues with codes 1, 2 and 3 rather than selecting or discarding trials to force the paper's reported count. Other annotation codes remain recorded in provenance and do not become task or rest labels.

There is no change to the dataset, participants, session partitions, channel selection, filtering, features, training labels, model, policy grid, scoring rules or stopping rules. Rest retains the same guarded protocol based definition. This amendment and its machine readable layout must be committed before fitting, and the runner verifies both files against the supplied amendment commit.

## Sources

The [original Figshare metadata](https://api.figshare.com/v2/articles/2061654) identifies the verified archive. The [MNE CNT reader documentation](https://mne.tools/stable/generated/mne.io.read_raw_cnt.html) describes the explicit data format and sample count reconstruction arguments. The [original study](https://doi.org/10.1371/journal.pone.0162657) describes the cue timing and recording procedure used to check the interpretation.
