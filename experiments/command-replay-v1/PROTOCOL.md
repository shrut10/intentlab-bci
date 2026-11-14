# Recorded commands on a separate EEG corpus

Protocol dated 14 November 2026, written before model fitting, policy selection or inspection of final session outcomes. This is an exploratory repository protocol, not an externally registered study.

## Question

Does a simple detector of task activity reduce unwanted commands during instructed rest compared with the same movement classifier using confidence alone? The experiment examines recorded command behaviour on a separate public corpus. It does not update the deployed IntentLab decoder or establish live headset performance.

## Source and chronology

Use the original Zhou2016 CNT archive from [Figshare](https://doi.org/10.6084/m9.figshare.2061654), whose source metadata states CC0. Verify the archive size and publisher MD5 given in `protocol.json`, and record its SHA256 locally. The later MOABB BIDS conversion has its own CC BY 4.0 record and is not the source used here. Publish code, provenance and derived decision tables, without redistributing the raw archive.

The [original paper](https://doi.org/10.1371/journal.pone.0162657) describes four experienced healthy participants, three sessions per person and two runs per session. Each run contains 75 imagery cues, with left hand, right hand and feet tasks. The 14 EEG channels were acquired at 250 Hz relative to the left mastoid, with acquisition filtering from 0.1 to 100 Hz and a 50 Hz notch.

For every participant, fit models using session 1, select the command policy using session 2, and evaluate the final fixed policy using session 3. Both runs remain in their assigned session. No overlapping windows may cross these partitions. This tests transfer to a later session of the same person, not transfer to unseen people.

## Signal and fitting

Select the 14 named EEG channels in `protocol.json`; exclude auxiliary channels. Re reference to their common average and apply a causal Butterworth bandpass with design order four from 8 to 30 Hz, retaining filter state throughout each run and resetting at run boundaries. Initial filter state is zero. Use trailing two second windows every half second, beginning with the first complete window. A decision time is its window end, and the endpoint at the exact recording duration is omitted.

Calculate Welch spectra with 250 samples per segment. For each channel, sum density bins in 8 to 13, 13 to 20 and 20 to 30 Hz, using inclusive lower and exclusive upper edges; take the logarithm with a floor of 1e-10 in squared microvolt units. Fit a StandardScaler and a balanced LogisticRegression with C equal to one, maximum 1000 iterations and seed 2027 separately for each participant. No additional hyperparameter search is permitted.

The movement classifier fits only complete windows within labelled left or right imagery. The task gate fits those same movement windows against eligible rest windows. Feet imagery is excluded from fitting. The two conditions are movement confidence alone and movement scores multiplied by the task probability. The products are decision scores and are not claimed to be calibrated joint probabilities.

## Timing labels

Imagery runs for five seconds from each labelled cue according to the source protocol. Eligible rest extends from 5.5 to 8.5 seconds after a cue, only when that complete interval lies within the file and ends no later than one second before the next cue. This buffer excludes the immediate task offset and the preparation period. These are protocol inferred periods of instructed relaxation, not direct measurements of the absence of intention.

Feet imagery is labelled as other activity for a secondary audit. Preparation, transitions and uncovered time are unknown. Events form an explicit contiguous partition of each recording and use half open intervals. Prediction is run throughout the recording; event labels are used for fitting eligible training windows and for scoring, never as a runtime trigger for prediction.

## Decisions and policy selection

Use the common IntentLab command simulator for both conditions. Enter thresholds are 0.5, 0.6, 0.7, 0.8 and 0.9, with required hold durations of zero, half a second or one second. Release threshold is 0.4, cooldown is 1.5 seconds and the maximum allowed gap between prediction rows is 0.75 seconds. The simulator defines latching, ties, release and gap behaviour consistently for both conditions.

For each condition, pool session 2 results and select the greatest fraction of left or right cues whose first command is correct, among policies producing no more than one command per minute of eligible rest. Break ties by lower idle command rate, then higher enter threshold, then longer hold. The validation target additionally requires a correct first command on at least half the imagery cues. If no candidate meets both requirements, use enter threshold 0.9 and a one second hold and explicitly report failure to meet the target. Feet outcomes do not select policies.

## Reporting and stopping

Report the selected policy, validation target result, all policy selection results, final session summary and every participant's results. Primary measures are the proportion of cues whose first command is correct and commands per observed minute of eligible rest. Also report correct command coverage, wrong hand commands, delay to the first correct command, unsupported activity commands, unknown period commands and cues receiving no commands. Describe the time denominator explicitly.

Do not change features, fitting rules, event intervals or policy choices after examining final session results. If the source layout is incompatible, stop and document the obstacle instead of silently substituting a dataset or changing labels. Four participants are insufficient for a broad population claim, so report their individual results without a bootstrap population interval. Any reduction in unwanted commands must be considered alongside missed intentions and delay.

The numerical settings in `protocol.json` are the executable specification. Preserve both protocol files after the initial commit. Any later correction must be recorded as an amendment, with its timing relative to outcome inspection.
