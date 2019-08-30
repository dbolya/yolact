# YOLACT Change Log

This document will detail all changes I make.
I don't know how I'm going to be versioning things yet, so you get dates for now.

```
2019.08.29
  - Fixed a bug where the fpn conv layers weren't getting initialized with xavier since they were being overwritten by jit modules (see #127)
2019.08.04
  - Improved the matching algorithm used to match anchors to gt by making it less greedy (see #104)
2019.06.27
  - Sped up save video by ~8 ms per frame because I forgot to apply a speed fix I applied to the other modes.
```
