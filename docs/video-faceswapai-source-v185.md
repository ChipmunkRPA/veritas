# Verified public synthetic face-swap source pair

The exact input MP4, production swap MP4 and synthetic donor JPEG from
FaceSwapAI's Controlled Production Video Face-Swap Study v1.0.0 were
downloaded locally. All three match the publisher's byte lengths and
SHA-256 values. The publisher states that both adult identities were
created synthetically and licenses the study under CC BY 4.0. The local
copy of the public JSON has SHA-256
`e73ecc30c700b95246fa56977e6e596eb685035a16a0b19d72e9f14e76095e8c`.
[Publisher study and rights](https://faceswapai.com/resources/video-face-swap-continuity-study) · [Versioned source manifest](https://faceswapai.com/video-continuity-study-v1.json)

Both MP4s contain only a video stream. The production probe decoded the
same 240 non-interpolated native picture timestamps for both clips at
1280 × 720. The time base is 1/12288, with each presentation instant
equal to its frame index divided by 24. The production decoder also
replayed all eight publisher sample positions in each file, with exact
index and presentation time correspondence.

The input declares full-range YUV and the output leaves color range
unspecified. Preserve that distinction and verify a common color and
encoding path before constructing a bounded insertion. Include an
unswapped control through the same encode and an ordinary recompression
control. The source pair is silent, its target is made from synthetic
still states, and the publisher swap covers the complete clip. This
qualifies the source for a controlled swap diagnostic. It supplies no
genuine-camera false-alarm estimate, lip-sync test or model score.
