# Filtering and recording time

This note, added on 10 November 2026, describes the existing preprocessing more precisely and documents additional checks for future data preparation. The [original protocol](PROTOCOL.md), processed data, trained models, calibration, thresholds and archived results remain unchanged.

## The filter used by the model

Each input contains 480 samples at 160 Hz, covering the interval from one to four seconds after an imagery cue. Common-average referencing uses the original 64 channels before retaining the nine model channels. In [`preprocess`](../src/intentlab/signal.py), each channel's mean is removed, the completed epoch is filtered, and one RMS value across channels and time normalises the result.

The implementation is `butter(4, [8, 30], btype="bandpass", fs=160, output="sos")`, followed by `sosfiltfilt`. SciPy's `N=4` specifies the Butterworth prototype order; the band-pass transformation produces an eighth-order filter represented by four second-order sections. Forward-and-backward application gives zero phase and a combined effective order of 16. The original protocol's “fourth-order” wording refers to the design parameter, rather than the order of the resulting two-pass band-pass operation. See the [SciPy Butterworth documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) and [forward-and-backward filtering documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html).

The two passes multiply the magnitude responses. At the design boundaries, 8 Hz and 30 Hz, the theoretical gain is approximately −6.02 dB after both passes, compared with −3.01 dB for one pass. This describes the filter before RMS normalisation; the final normalised values do not preserve an absolute voltage gain. The 8–30 Hz description is therefore a filter specification, not a claim that all frequencies inside those limits pass unchanged or that all frequencies outside them disappear.

Filtering is applied independently to each epoch. [`sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) uses its default odd-extension padding at the window boundaries. Finite-window edges can still influence the filtered features, and zero-phase filtering can spread a transient both before and after its original time. The processed trace should not be used to infer the precise onset of a neural event. No filter design, padding or normalisation setting has been changed for this clarification.

## Completed windows and live processing

The current filter uses later samples within each completed window. Mean removal and RMS normalisation also require that whole window. These operations do not mix participants or borrow data from another epoch, but they cannot produce the same output immediately as each individual sample arrives.

A system could process a completed trailing window once its final sample becomes available; that design must include the window's collection time in its latency assessment. In the current cue-aligned protocol, the input ends four seconds after the cue. The browser replays already processed recordings, so API processing time does not measure the time from an intention to a live command.

A continuously updated version would need a separately evaluated preprocessing and decision pipeline. A causal filter must retain its state between chunks and have explicit warm-up and reset behaviour after a gap. Replacing `sosfiltfilt` alone would not resolve the whole-window scaling, timing or model-distribution questions. Those changes require a new experiment and do not belong in an unannounced update to the deployed model.

## Recording continuity checks

Future preparation validates the selected source files before extracting epochs or writing derived outputs. The supported input is the continuous EDF+ subset used by this corpus: EDF+C, a known positive record count and duration, and a first record starting at offset zero. Fractional initial offsets are valid in EDF+ generally, but are deliberately rejected here because the current cue-to-sample conversion assumes a zero origin.

The preflight checks the header structure, declared samples and file size, and the first timekeeping annotation in every record of the first EDF Annotations channel. Successive onsets must match the declared record duration exactly. Missing or malformed timekeeping annotations, gaps, overlaps, repeated or out-of-order record onsets, and incomplete or extra record bytes cause preparation to stop with an error. EDF+D and ordinary EDF without the required timekeeping metadata are unsupported. The checks do not repair, interpolate or silently join recordings across a gap; an unexpected file is an acquisition failure, not an additional participant exclusion.

These requirements are based on the [EDF+ specification](https://www.edfplus.info/specs/edfplus.html), which distinguishes continuous EDF+C from discontinuous EDF+D and records the onset of each data record. The checks verify the timing information stored in a file. Neither these checks nor publisher checksums can establish that the acquisition equipment never lost samples before producing that file. Checksums establish that the downloaded bytes match the published source.

With the source files cached locally, run the read-only audit without regenerating data or models:

```sh
python scripts/audit_edf_timing.py
```

The new checks are additional ingestion safeguards introduced after the original experiment. They must not be described as part of its prespecified eligibility rules or as evidence of a prospective headset evaluation.
