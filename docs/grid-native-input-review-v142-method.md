# Independent replay of the complete saved GRID input bank

Freeze this new context only after all sixteen original archive acquisitions and
all sixteen v137 native-input stages have actual complete receipts. Admission
reads and hashes code, frozen plans, JSON journals, receipts and resource profiles
only. It does not read PNG, WAV, NPY, MPG or whole archive content. Preserve the
original failed v131 stage: its sole complete first archive is admitted through
the successful independent v134 full-archive preflight. Sources 01–15 require
their own successful v135 acquisition receipts and archive journals. Never turn
the stopped v131 stage into a successful acquisition receipt.

Independently join every directory, full archive record and selected member.
Reconstruct v136's fixed salted utterance-stem ordering across all sixteen slots;
require exactly its two original members per slot, 32 complete ordered records,
and the previously fixed first-source choices. Bind all native-input JSONs and
all successful receipt/profile/resource-journal hashes. Inspect actual clean
samples and unchanged limits; a verified terminal process transition is allowed,
but missing receipts, resource violations and sampling failures are not.

After publishing the freeze, run one exclusive `bind-00` data worker through the
unchanged v108 owned-descendant monitor: 900 seconds, 1.5 GiB sampled descendant
RSS, 3 GiB initial available RAM/project/archive disk and 1.5 GiB running reserves.
Admission denial creates no attempted stage; once attempted, preserve any failure
without retry, overwrite, substitution or resource relaxation. The worker must:

- Hash every saved MPG, verifying its complete size and independently computed
  ZIP CRC against the exact originally selected member. Do not decode the MPG or
  reopen whole ZIP archives. Earlier full archive verification is receipt-bound,
  not independently repeated by this review.
- Verify all saved PNG file hashes; independently use Pillow to decode RGB bytes
  and compare their raw-byte hashes. Check dimensions, complete native frame-index
  order and exact observed PTS against the saved video table. Use `Fraction` for
  time bases and origins; do not call application frame/clock helpers or infer
  timestamps from frame index/frame rate. All 2,400 observed picture records must
  be retained when the frozen bank contains that count.
- Independently read canonical WAV using Python `wave`: one channel, 16 kHz,
  signed 16-bit uncompressed samples. The exact signed-sample/32768 conversion must
  equal the saved finite little-endian float32 NPY values and the mapping's PCM
  hash. Reject a matching outer file hash with different PCM values.
- Reconstruct complete sample ranges without gaps, overlap or source-clock jumps.
  Rebuild native audio intervals from recorded integer PTS, rational time base and
  per-frame sample count; independently verify their stored intervals and flags.
  Native continuity allows only the original one timestamp tick plus one native
  audio sample. Canonical endpoints allow that quantization plus one 16-kHz
  sample. Preserve separately observed audio/video origins, including nonzero
  origins; do not silently reset clocks or extend beyond recorded support.
- Retain every unavailable status and its saved partial observations. Unavailable
  is not an original-file verdict; no case is dropped or replaced. The current
  completed bank is observed available on all 32 clips; the review must verify
  that metadata commitment and actual content, not assume it from progress text.

Privately group exact complete decoded-picture sequences, exact PCM content and
source-MPG bytes, then join those links with the two clips' original source slot.
Missing content never forms an equal-content group. Save private membership and
per-input commitments; publish aggregate duplicate/component counts only. These
are known content/source-slot links, not proof of distinct people, unknown
recording ancestry or pretrained-model independence.

Publish code, tests, aggregate counts and whole-artifact commitments only. Keep
all raw media, per-media hashes, source/member identities and pairings, clocks,
vectors and private journals out of the public package. This separate replay
checks saved-bank consistency; it does not reproduce original MPEG decoding,
verify source authenticity, execute a model/encoder, establish achieved lip sync,
measure detection accuracy or change customer confidence. No Actions workflow,
outreach, payment, deployment or serving-default change is included.
