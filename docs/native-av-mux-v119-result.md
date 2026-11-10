# Native audiovisual assembly: actual result

All three cases in the frozen [assembly procedure](native-av-mux-v119-method.md)
passed: two authored timing/sample fixtures and one complete first-fitting-source
rewrite. All **76 picture timestamps and RGB images** and **43,912 signed-16 audio
samples** match after output decoding. Audio origins and continuous source/sample
mappings are unchanged, including the 42-ms audio/67-ms picture offset fixture
and a partial final audio packet.

The stage made no neural calls. It exited zero in 3.72 seconds, with no resource
guard or sampling error. Maximum sampled descendant RSS was 231,325,696 bytes;
minimum sampled available RAM was 7,513,522,176 bytes. These samples are not true
peaks. Actual receipt, resource-journal and private full-case hashes agree in the
[public execution record](evaluation-panels/native-av-mux-result-v119.json).

The runtime is isolated from the application environment. Its publisher wheel,
installed source/binary hashes and bundled codec versions are recorded. PyAV's
bundled FFmpeg and the system command decoder are separate builds in the same
codec lineage, not independent codec implementations. The output's last-picture
duration is constructed from the upper median observed step; no original
exposure time or terminal display-duration equivalence is claimed.

Output clips and individual source/frame/audio fingerprints remain private on
Seagate. This result qualifies the complete-picture/PCM assembly path for known
native mouth-edit construction. It does not generate a new edit, measure achieved
synchronization, evaluate a detector or change customer confidence. Wider edited
clips, ordinary controls and the articulation comparison remain unfinished.
No outside contact, Actions workflow, real charge or GCP deployment occurred.
