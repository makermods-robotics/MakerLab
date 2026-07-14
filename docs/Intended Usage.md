Intended Usage:

1. 2 ports and up to 2 cameras will be on a single port that describes a leader-follower pair.
2. Up to 2 of these pairs will be used at once.
3. Not more than 3 cameras will be used.
4. During a session of usage with this UI, the every USB will remain plugged in. Between sessions, the arrangements of ports may change.
5. Main is the only stable and end-to-end hardware-tested branch.
6. While using a Jetson, the user will have a display connected to the Jetson, as well as a keyboard/mouse.
7. This UI will only be used with human supervision.
    For now, edge case bugs exists when the UI accepts mismatched calibration files to arms. In this case, teleoperation will put joints in potentially out-of-bounds/harmful positions.
8. The user does not need to log into HuggingFace, but this disables the ability to upload datasets and models, as well as the ability to train on HF Jobs.
9. This UI will only be used with SO-101 arms.
10. This UI cannot be run twice simultaneously
