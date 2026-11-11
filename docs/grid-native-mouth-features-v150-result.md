# Complete native feature bank for the 176-case pilot

The sixteen guarded feature stages completed all **176 matched cases**. All
**13,200 decoded picture positions** had a usable native mouth region. The
shared audio support covered **12,320 positions**; 880 lacked complete audio
context and stayed outside that scoring scope. The extraction did not use
known edit labels, bounds or masks to choose features.

Exact byte matches avoided repeated image inference at 9,440 picture positions.
The workers recorded 3,776 face inputs and 3,776 image encoder inputs, including
16 fixed original repeats that matched exactly. The recorded output timestamps
and each case's actual whole-waveform speech states remained attached. No new
speech encoder ran here. Original and audio-delay controls share visual data but
use separately encoded soundtracks.

All stages exited zero, with no guard or sampling errors. Summed worker-stage
time was 397.95 seconds; maximum sampled descendant RSS was 0.95 GiB. See the
[actual receipt](evaluation-panels/grid-native-mouth-features-complete-v150.json)
and [frozen method](grid-native-mouth-features-v150-method.md).

This is the complete blind feature bank for the locked comparison. It is not a
detector accuracy or achieved synchronization result. The reserve was not used.
All vectors, raw media, per-input clocks and individual fingerprints remain
private; only aggregate counts and whole-artifact commitments are published.
