# An open intracortical dataset for a future drift study

Neuralink's [October 2026 article](https://neuralink.com/updates/pretraining-on-50000-hours/) discusses recordings collected during implanted-BCI use, but does not release the reported 50,000 hours or trained encoders. Its reference to NoMAD leads to a separate data deposit with publicly accessible metadata and a reusable licence.

[Karpowicz et al. (2025)](https://doi.org/10.1038/s41467-025-59652-y), *Stabilizing brain-computer interfaces through alignment of latent dynamics*, provides [two human recordings on Dryad](https://doi.org/10.5061/dryad.q83bk3jtp). The versioned metadata declares CC0-1.0 and publishes a SHA-256 and byte count for each recording. This repository records those expected values in [the metadata manifest](../artifacts/external-data/nomad_manifest.json).

## What is available and what has been verified

**Metadata verified; recording download blocked.** On 26 October 2026 the metadata API was accessible, but its download endpoint returned HTTP 401 and the dataset page's public download link returned HTTP 403. The API labelled version 5 as `submitted`; the cause of the download restrictions is not established. We did not bypass these responses. No recording bytes were obtained, no real-file inspection was completed, and no model was trained on this dataset. The manifest therefore leaves computed checksums and inspection results null. Published expected checksums are not presented as locally verified checksums.

The two NWB recordings total 247,179,216 bytes (about 247 MB). The dataset also lists a separate 1,820-byte README. Its documentation describes one BrainGate participant, T11, with 192 intracortical recording channels aggregated into 20 ms spike-count bins. Published fields include cursor and target position, cursor velocity, recording-block identifiers and trial calibration flags. These descriptions are **publisher-reported**, rather than observations from downloaded arrays.

This is not scalp EEG, not Neuralink's dataset and not an independent replication of IntentLab's left/right motor-imagery experiment. It must remain a separate study with its own protocol. The published task includes sequential target recall, and the accompanying methods construct intended trajectories using target information and select successful movements. Actual cursor movement, instructed target and inferred intention should not be treated as interchangeable labels.

## Inspect the saved metadata or prepare a future download

Run from the repository root. Printing the checked manifest needs only Python's standard library and makes no network request:

```sh
python examples/nomad_data_audit.py
```

If the publisher's public downloads become accessible, this command downloads only the two named files, enforces their byte limits and verifies both published SHA-256 values before accepting the files:

```sh
python examples/nomad_data_audit.py --download
```

The downloader stops on an access failure, corruption or excess bytes. It does not seek credentials, change access methods or silently accept an incomplete file. The default destination is ignored `data/raw/nomad/`; raw participant recordings are not committed to GitHub or included in the web deployment.

Once recordings are legitimately available, inspect their actual sample, trial and channel counts and timing in a separate environment:

```sh
python3 -m venv /tmp/intentlab-nomad-audit
/tmp/intentlab-nomad-audit/bin/python -m pip install h5py==3.16.0
/tmp/intentlab-nomad-audit/bin/python examples/nomad_data_audit.py \
  --inspect --output /tmp/intentlab-nomad-inspection.json
```

Inspection verifies the recording checksums first, then reads acquisition shapes, spike timing, trial counts, calibration flags and recording-block IDs. It is a structural audit, not complete NWB schema validation or a predictive experiment. The optional HDF5 reader is not a production dependency. Unit tests cover checksum failure and download boundaries without fetching real recordings; an additional small synthetic NWB-shaped fixture exercises the inspector when `h5py` is installed.

## What this could support later

A separate intracortical study could compare a frozen decoder across recording blocks while measuring adaptation burden. Any such study needs prespecified chronological splits, a clear account of which neural and behavioural information enters fitting, and uncertainty appropriate to a single participant. The current contribution is reproducible data-ingestion preparation with explicit access status, not a decoding result or a reproduction of NoMAD.

Sources: [dataset metadata](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.q83bk3jtp), [versioned file metadata](https://datadryad.org/api/v2/versions/355813/files), [data and task documentation](https://doi.org/10.5061/dryad.q83bk3jtp), [peer-reviewed study](https://doi.org/10.1038/s41467-025-59652-y).
