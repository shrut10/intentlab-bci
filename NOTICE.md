# Data attribution and notices

This project uses the **PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0**, contributed by Gerwin Schalk and colleagues at the BCI R&D Program, Wadsworth Center, New York State Department of Health.

Source: https://physionet.org/content/eegmmidb/1.0.0/

DOI: https://doi.org/10.13026/C28G6P

Licence: **Open Data Commons Attribution License v1.0 (ODC-BY 1.0)**

Licence text: https://opendatacommons.org/licenses/by/1-0/

`data/demo/epochs.npz`, `data/demo/trials.json`, `data/derived/trials.parquet`, and the source-record entries in `data/derived/manifest.json` are derived from this database. Their attribution and source licence must be retained when shared. The raw EDF files are downloaded from the official public mirror, verified against the publisher's checksums, and remain outside Git.

Changes made: common-average referencing, selection of nine channels, extraction of 1–4 second imagery intervals, deterministic quality exclusions, 8–30 Hz filtering, per-epoch RMS normalisation, bandpower feature extraction, subject-level partitioning and selection of the first six valid test epochs per participant for the demo. These derivatives are not endorsed by PhysioNet or the original investigators.

## Requested source citations

1. Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C28G6P
2. Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043. https://doi.org/10.1109/TBME.2004.827072
3. Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*, 1, 792–795. https://doi.org/10.1038/s44360-026-00096-z

The CNN's architectural inspiration is Lawhern et al. (2018), *EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces*, https://doi.org/10.1088/1741-2552/aace8c. IntentLab implements an adapted small network with odd temporal kernels, adaptive average pooling and a single binary output; it does not claim to reproduce the paper's architecture or results.

The MIT licence in this repository covers original application code only. Third-party dependencies retain their own licences.
