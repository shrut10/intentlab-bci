# Data card

## Source and access

[PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), DOI 10.13026/C28G6P. Open access under ODC-BY 1.0. Attribution and original-study citations are in `NOTICE.md`.

The original dataset contains scalp EEG recorded during several executed and imagined movement tasks. This experiment uses **only runs 4, 8 and 12**, the left/right imagined-fist task. T1 means left fist and T2 means right fist in these runs. The same annotation codes have different meanings in other runs; mixing them would create incorrect labels. Rest annotations are not classified.

The primary source was used rather than an unverified re-upload. A Hugging Face account or dataset mirror is not required.

## Published artifacts

| Artifact | Contents |
|---|---|
| `data/derived/manifest.json` | All 327 source URLs, file sizes, checksums, split IDs, preprocessing metadata and exclusions |
| `data/derived/trials.parquet` | One row per eligible epoch: subject/run/event IDs, label, split, cue onset, source URL, amplitude audit and 27 log-bandpower features |
| `data/demo/trials.json` | Metadata for the first six valid test epochs of each retained test person |
| `data/demo/epochs.npz` | The corresponding 126 × 9 × 480 processed float32 signals, loaded without pickle |
| `data/raw/` (ignored) | Original EDF downloads and publisher checksum list |
| `data/derived/epochs.npz` (ignored) | Full processed training/validation/test tensors, reproducible from the source |

The public CSV download contains the clean, processed epoch, not synthetic disturbances applied in the interface. Values are normalised amplitude, not raw microvolts. The original EDF is linked from the metadata. The nine electrodes still require all 64 channels for the common-average reference in this protocol; this does not establish performance using a nine-electrode physical headset.

## Predefined eligibility checks and observed exclusions

- S088, S092 and S100: all three selected recordings at 128 Hz, rather than the required 160 Hz. No resampling was added after seeing the data.
- S099-R04-E05: peak-to-peak amplitude exceeded 1,000 microvolts in a selected, referenced channel.
- S104-R08-E25: the labelled interval did not cover the complete 1–4 second epoch.
- No other eligibility exclusions occurred.

Acquisition failures abort the pipeline; they are not silently counted as quality exclusions. All 327 source files passed publisher-checksum verification.

| Split | People after checks | Trials | Left | Right |
|---|---:|---:|---:|---:|
| Training | 63 | 2,832 | 1,430 | 1,402 |
| Validation | 22 | 989 | 498 | 491 |
| Test | 21 | 945 | 476 | 469 |

Membership is fixed by a seed-2027 permutation of all 109 source IDs before exclusions. No person appears in more than one split. No personal, employer or private user data is included.

## Interpretation limits

Public availability does not turn EEG into a general thought-reading resource. Labels describe the experimental cue, not independently observed thought content. The analysis has not established demographic representativeness, clinical transfer or performance on a new headset. The app never infers identity, diagnosis, dreams or consciousness.
