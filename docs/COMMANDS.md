# Auditing commands from EEG decoder scores

IntentLab can replay timestamped decoder scores and measure the commands that a fixed controller would have issued. This makes it possible to inspect missed cues, incorrect directions, repeated commands and commands during annotated rest, alongside the amount of recording available for evaluation. It is a local analysis tool for recorded scores, with no EEG upload and no device connection.

The decoder and controller have different jobs. A decoder estimates class probabilities from an EEG window. The controller decides whether those scores have remained strong enough for long enough to issue a command. A model with reasonable trial accuracy can still produce unwanted commands when it runs continuously, particularly between cues. Conversely, a strict controller can avoid most commands while missing many intended actions. Both behaviours need to be visible.

This module does not train a model, calibrate probabilities or choose thresholds. It does not change IntentLab's deployed model. Its reports describe replay outcomes, not demonstrated control by a person using a live headset.

## Try the example locally

From the repository root, run:

```sh
python examples/command_audit.py \
  --predictions examples/command_audit_sample/predictions.csv \
  --events examples/command_audit_sample/events.csv \
  --policy examples/command_audit_sample/policy.json \
  --output /tmp/intentlab-command-example.json
```

The example contains **four SYNTHETIC score rows written by hand**. They are not recorded EEG, model predictions or evidence of performance. The participant identifier is `SYNTHETIC` to keep this distinction explicit in the exported report.

The first two rows prefer left at times 0 and 0.5 seconds, causing one left command at 0.5 seconds. The next two prefer right at 1 and 1.5 seconds, causing one right command at 1.5 seconds. The annotations contain a left cue from 0 to 1 second and rest from 1 to 2 seconds. The expected report therefore contains one correctly served cue, a first correct latency of 0.5 seconds, and one command during one second of annotated rest. The latter is 60 commands per minute after converting the denominator; such a short synthetic interval provides no useful estimate of real reliability.

The output must be a new file. The command refuses to overwrite an existing report, including the example path above on a second run. Use a different output path when comparing policies. Input CSV files are limited to 20 MB each.

The report includes SHA256 hashes of the exact prediction and annotation CSV bytes. These identify the supplied inputs; they do not establish that the inputs are genuine recordings. The validated policy is also stored in the report. No source data is sent to a server.

## Input contract

### Prediction CSV

Each row has these columns:

| Column | Meaning |
| --- | --- |
| `participant` | Participant identifier |
| `session` | Session identifier within that participant |
| `time_s` | Time in seconds from the start of the session |
| `left_probability` | Decoder score for left, between 0 and 1 |
| `right_probability` | Decoder score for right, between 0 and 1 |

Despite their column names, these values can be probabilities or decision scores that satisfy the same range and sum constraints. The tool does not certify calibration. For example, the recorded study's rest gate multiplies the score for movement by the score for a direction; this is a decision score and is not established as a calibrated joint probability.

The two values must sum to at most 1, allowing any remaining score to belong to another class or rejection. A small tolerance of 0.000000001 permits floating point rounding. Scores and times must be finite. Each participant and session pair must have strictly increasing nonnegative timestamps in the supplied row order. Rows from separate sessions can be interleaved; the audit does not silently sort or repair a session.

Use a timestamp for when a prediction would actually be available, such as the end of its EEG window. Do not use the centre or beginning of a window while claiming that the result was available then. Processing latency, transport latency and actuator latency are not automatically added by this tool. If your model or preprocessing accesses future samples, the audit cannot turn those inputs into evidence of causal processing.

Identifiers are strings or integers that become strings in the output. They must contain 1 to 128 characters, with no surrounding spaces or control characters. The prediction list must contain at least one row.

The simulator reads only the five prediction fields above. It never reads cue annotations, class labels or future rows. Additional prediction fields are ignored. Labels are supplied separately to the audit after commands have been simulated.

### Event CSV

Each row has these columns:

| Column | Meaning |
| --- | --- |
| `participant`, `session` | The same identifiers used in the prediction CSV |
| `event_id` | Unique event identifier within the session |
| `start_s`, `end_s` | Event boundaries in seconds |
| `state` | One of `left`, `right`, `idle`, `other` or `unknown` |

Intervals include their start and exclude their end, so a command exactly at a boundary belongs to the next event. Each interval must have positive duration. Within each session, intervals must be ordered and exactly contiguous, starting at zero and ending at the declared end of that session. Every prediction must fall inside this coverage. The two inputs must contain the same participant and session pairs.

Use `idle` only for a period you intend to evaluate as rest or no requested control. Use `other` for a known activity outside the two supported commands, and `unknown` when the activity cannot be established. Keep these states separate rather than silently treating every missing task annotation as rest. A gap in event coverage is rejected; an explicitly labelled `unknown` interval is accepted.

Cue boundaries are used only for scoring. They do not reset the controller or tell it when to issue a command. A candidate that began before a cue can consequently produce a command inside that cue. This is one reason a short measured latency is not, by itself, evidence of a response to the cue.

### Policy JSON

The policy must contain exactly these five fields:

| Field | Constraint | Meaning |
| --- | --- | --- |
| `enter_threshold` | 0.5 to 1 inclusive | Minimum score for the unique winning direction |
| `release_threshold` | At least 0 and strictly below entry | Score at which an already emitted direction is released |
| `hold_seconds` | At least 0 | Required duration of consecutive qualifying observations |
| `cooldown_seconds` | At least 0 | Minimum interval after an emitted command before a new hold can begin |
| `max_gap_seconds` | Greater than 0 | Largest permitted interval between adjacent observations |

When `--policy` is omitted, the CLI uses entry 0.7, release 0.4, hold 0.5 seconds, cooldown 1.5 seconds and maximum gap 0.75 seconds. The synthetic example supplies a shorter cooldown of 0.5 seconds so that its arithmetic is easy to follow. Defaults are an explicit starting policy, not a scientifically established operating point.

Fix policies before evaluating a final test set. If selecting a policy from data, use a separate validation set and preserve the selection rule and chosen values. This tool deliberately performs no optimisation and offers no guarantee that a threshold will meet a reliability target on new people or recordings.

## Exact controller behaviour

1. A candidate begins on the first row whose unique winning direction reaches the entry threshold. Equal scores abstain, including a 0.5 versus 0.5 tie.
2. Further qualifying rows for the same direction continue its hold. A change of direction, a tie or a score below entry resets the candidate. No values are inferred between observations.
3. The first qualifying observation at or after the hold duration emits one command. Its confidence is the winning score on that row, not a guarantee that the command is correct.
4. The emitted direction remains latched until its own score reaches or falls below release. Repeated strong observations alone cannot emit repeated commands. An opposite strong direction can release the old latch and begin a new candidate on that same row when cooldown permits.
5. Rows during cooldown cannot contribute to a new hold. The first eligible row after the deadline starts a fresh candidate, even if earlier rows favoured the same direction.
6. A gap strictly greater than `max_gap_seconds` resets the candidate and latch, while preserving an outstanding cooldown deadline. Equality does not create a gap. Session changes start independent controller state.

Clock comparisons allow two floating point units of rounding error at a timing boundary. This avoids treating decimal representations such as 0.3 and 0.1 + 0.2 as meaningfully different observation times. Timestamps must nevertheless remain strictly increasing.

## Report schema and interpretation

The JSON report has `schema_version: "1.0"` and `kind: "command_audit"`, followed by `policy`, `summary`, `participants`, `sessions`, `commands`, `gaps` and `event_results`. The CLI adds `provenance`. Summary metrics pool all supplied sessions. Participant and session entries use the same metrics and retain groups that produced no commands.

### Cue and command metrics

| Field | Definition |
| --- | --- |
| `prediction_rows` | Number of score observations supplied |
| `command_count` | Total emitted commands |
| `cues` | Number of left or right event intervals |
| `cues_with_correct` | Cues containing at least one correct direction command |
| `cues_first_command_correct` | Cues whose first emitted command has the correct direction |
| `missed_cues` | Cues containing no correct command, including cues with only wrong commands |
| `cues_without_commands` | Cues containing no command of either direction |
| `correct_commands` | Commands with the requested direction during a hand cue |
| `wrong_hand_commands` | Commands with the opposite direction during a hand cue |
| `repeated_correct_commands` | Correct commands after the first correct command within the same cue |
| `idle_commands` | Commands during annotated idle intervals |
| `other_commands` | Commands during known activity outside the two directions |
| `unknown_commands` | Commands during intervals with unknown activity |

A cue with a wrong first command and a later correct command increases `cues_with_correct`, but not `cues_first_command_correct`. Report these alongside missed cues and repeated commands so that eventual success does not conceal an initially wrong action.

`first_correct_latencies_seconds` contains one latency per cue with a correct command, measured from cue start to its first correct command. `latency_observations` states how many cues contributed. `median_first_correct_latency_seconds` is their median, or `null` when there are none. Missed cues are never assigned zero latency. These are conditional latencies and must be read alongside the total cue count and missed cues.

Every item in `commands` contains participant, session, time, direction and confidence, together with its `event_id`, annotated `state` and `outcome`. Outcomes are `correct`, `wrong_hand`, `idle`, `other` or `unknown`. Every item in `event_results` retains the annotation and includes command counts, correct and wrong direction counts, repeated correct counts, `first_command_correct` and `first_correct_latency_seconds`. For hand cues, a missing first command or correct latency is `null`. For idle, other and unknown intervals, direction correctness and cue latency do not apply, so these two fields remain `null` even if commands occurred.

### Exposure, gaps and denominators

| Field | Definition |
| --- | --- |
| `idle_seconds` | Full duration of annotated idle intervals |
| `idle_commands_per_minute` | `60 * idle_commands / idle_seconds`, or `null` when no idle time is supplied |
| `annotated_seconds` | Full duration of all event intervals |
| `observed_span_seconds` | Sum of the interval from first to last prediction in each session |
| `gap_seconds` | Total duration of flagged intervals between observations |
| `idle_gap_seconds` | Duration of those gaps that overlaps annotated idle |
| `evaluated_span_seconds` | Observed spans minus flagged gaps |
| `unobserved_edge_seconds` | Annotated duration before the first and after the last prediction in each session |

A reported gap covers the whole interval between two adjacent rows when that interval exceeds the permitted maximum. It is not an estimate of how many individual packets were lost. There is no gap before the first observation or after the last; those periods are represented by `unobserved_edge_seconds`. Each gap stores participant, session, start, end and duration in seconds.

The idle command rate deliberately uses **all annotated idle time**, including outages and unobserved edges. Gaps are not filled and do not invent commands. A long outage can therefore make this rate look lower. Inspect `idle_gap_seconds`, `gap_seconds` and `unobserved_edge_seconds` alongside the rate, and do not interpret a low rate as proof of reliable continuous control. `evaluated_span_seconds` is an elapsed time descriptor between observed scores; it does not prove that every sample in that span was recorded or processed correctly.

Pooling idle counts and duration produces the summary rate. It is not an unweighted average of participant rates. Inspect participant and session results as well as the summary because a pooled result can conceal individuals with no useful commands or unusually frequent unwanted commands.

## Reuse from Python

```python
from intentlab.commands import build_command_audit, simulate_commands

# predictions and events are lists of dictionaries using the columns above.
# policy is a dictionary containing the five required policy fields.
replay = simulate_commands(predictions, policy)
report = build_command_audit(predictions, events, policy)
```

`simulate_commands` returns only the validated policy, commands and gaps. It has no annotation argument. `build_command_audit` applies the same simulator, then grades its output against events. Neither function mutates the supplied rows or policy. The module uses the Python standard library and can analyse saved outputs from a different decoder without loading IntentLab's model or the web API.

For a study using recorded data, preserve the score generation code, timing convention, class mapping, preprocessing and split rules as well as the exported inputs. CSV hashes alone cannot establish independence of training and evaluation data or rule out access to future samples.

## Research context and limits

Carrara and Papadopoulo (2023) describe evaluating recorded EEG through successive windows while accounting for idle activity, rather than limiting analysis to selected task epochs. Their work motivates examining the time surrounding cues as well as the cues themselves. This audit is a separate command level tool and does not claim to reproduce their metrics or benchmark. [Read the paper](https://arxiv.org/abs/2308.11656).

Liu, Li, Tan and Wu (2026) study distinguishing rest, supported motor imagery and activity outside the supported classes in continuous EEG. Their preprint reinforces the distinction between choosing a direction and deciding whether a control action should happen at all. IntentLab's fixed controller does not implement or validate their proposed detection method. [Read the preprint](https://arxiv.org/abs/2605.01014).

The independent recorded study is documented in [the command replay protocol](../experiments/command-replay-v1/PROTOCOL.md). Where rest is inferred from the intervals outside dataset cues, it remains an annotation convention rather than a direct measurement of a person's intention. Absence of a cue does not prove absence of imagery, movement, distraction or artefact. The recorded study and this reusable auditor therefore support inspection and comparison, while prospective work with people and suitable hardware is still required to assess actual control.
