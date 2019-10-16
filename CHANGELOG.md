# YOLACT Change Log

This document will detail all changes I make.
I don't know how I'm going to be versioning things yet, so you get dates for now.

```
Pending Changes:
  - Changed the default behavior for --start_iter from 0 to -1 (latest iter)

2019.09.20
  - Fixed a bug where custom label maps weren't being applied properly because of global default argument initialization.
2019.08.29
  - Fixed a bug where the fpn conv layers weren't getting initialized with xavier since they were being overwritten by jit modules (see #127).
2019.08.04
  - Improved the matching algorithm used to match anchors to gt by making it less greedy (see #104).
2019.06.27
  - Sped up save video by ~8 ms per frame because I forgot to apply a speed fix I applied to the other modes.
```
