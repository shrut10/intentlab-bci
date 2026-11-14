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

The follow-up in `artifacts/adaptation-v1/` derives trial identifiers, labels, probabilities and participant-specific transforms from the same source dataset, retaining the attribution above. It uses regularised Euclidean alignment inspired by He, H. and Wu, D. (2020), *Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach*, DOI https://doi.org/10.1109/TBME.2019.2913914, and temperature scaling following Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html. Neither reference endorses this implementation or its results.

The MIT licence in this repository covers original application code only. Third-party dependencies retain their own licences.

## Zhou2016 command study

The separate experiment in `artifacts/command-replay-v1/` uses the original CNT archive associated with Zhou, B., Wu, X., Lv, Z., Zhang, L. and Guo, X. (2016), *A Fully Automated Trial Selection Method for Optimization of Motor Imagery Based Brain-Computer Interface*, [PLOS ONE, 11(9), e0162657](https://doi.org/10.1371/journal.pone.0162657).

The [original Figshare release](https://doi.org/10.6084/m9.figshare.2061654) declares CC0 in its [source metadata](https://api.figshare.com/v2/articles/2061654). The study uses that release, rather than the later BIDS conversion with a separate licence. The raw archive remains outside Git; published derivatives contain event intervals, decision scores, command logs and aggregated results. Instructed rest intervals are inferred from the published protocol with explicit buffers, rather than supplied as direct measurements of intention. The original investigators do not endorse this analysis.

Changes comprise selecting the 14 EEG channels, common average referencing, causal filtering, trailing window features, participant specific fitting on the first session, policy selection on the second session and fixed evaluation on the third session. Provenance records the source checksums, protocol commit, software and output hashes. The original PhysioNet model and its attribution remain separate.

## Dreyer2023 participant study

The separate experiment in `artifacts/unseen-v1/` uses recordings published by Pauline Dreyer, Aline Roc, Léa Pillette, Sébastien Rimbert and Fabien Lotte, described in [Scientific Data, volume 10, article 580](https://doi.org/10.1038/s41597-023-02445-z) and deposited under [DOI 10.5281/zenodo.8089820](https://doi.org/10.5281/zenodo.8089820)

The recordings are distributed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/); this study uses the MOABB EDF conversion hosted on OSF and mirrored by EEGDash on Hugging Face, with a pinned revision and checksums matched against the original conversion host for all 87 participant archives

Changes comprise retaining the first two acquisition runs, extracting delayed imagery windows, common average referencing over 27 EEG channels, selecting nine central channels, filtering and normalising completed epochs, deriving band power and covariance features, fitting separate fixed classifiers and recording probabilities and participant statistics

Published derivatives contain trial identities, labels, predictions and aggregated results; raw EEG, feature caches and fitted study models remain outside Git and deployment, and the source attribution must be retained with shared derivatives

Neither the dataset authors nor its distribution hosts endorse this analysis, which is separate from the original PhysioNet decoder and its licence notice above
