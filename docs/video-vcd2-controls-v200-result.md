# Twelve matched webcam control sets verified

All 36 controls completed locally from 12 camera recordings. The production
decoder verified 10,158 output pictures across the three roles, preserving
all 3,386 source-relative picture timestamps in each role.
[Construction method](video-vcd2-controls-v200-method.md) · [Verified coverage](evaluation-panels/video-vcd2-controls-complete-v200.json)

The unchanged copies preserved every normalized RGB picture. The chroma
and recompression controls each changed 999 decoded pictures, all within the
planned 3–6 second interval. All 36 inputs are camera-only controls, so any
face-swap review interval on them is a false flag for this study.

Thirty files fit the standard 500 MiB upload limit. Six exceed the limit
and stay in the raw study denominator as unsupported uploads. A separate
delivery encode tests practical upload versions of all 36 files. It checks
which paired pixel changes survive the encode and verifies the native clock.
[Raw full-app comparison](video-vcd2-full-app-v202-method.md) · [Delivery comparison](video-vcd2-delivery-v206-method.md)
