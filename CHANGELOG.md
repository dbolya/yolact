# YOLACT Change Log

This document will detail all changes I make.
I don't know how I'm going to be versioning things yet, so you get dates for now.

```
2019.11.06:
  - Changed Cython import to only active when using traditional nms.

2019.11.04:
  - Fixed a bug where the learning rate auto-scaling wasn't being applied properly.
  - Fixed a logging bug were lr was sometimes not properly logged after a resume.

2019.10.25 (v1.1):
  - Added proper Multi-GPU support. Simply increase your batch size to 8*num_gpus and everything will scale.
    - I get an ~1.8x training speed increase when using 2 gpus and an ~3x increase when using 4.
  - Added a logger that logs everything about your training.
    - Check the Logging section of the README to see how to visualize your logs. (Not written yet)
  - Savevideo now uses the evalvideo framework and suports --video_multiframe. It's much faster now!
  - Added the ability to display fps right on the videos themselves by using --display_fps
  - Evalvideo now doesn't crash when it runs out of frames.
  - Pascal SBD is now officially supported! Check the training section for more details.
  - Preserve_aspect_ratio kinda sorta works now, but it's iffy and the way I have it set up doesn't perform better.
  - Added a ton of new config settings, most of which don't improve performance :/

2019.09.20
  - Fixed a bug where custom label maps weren't being applied properly because of global default argument initialization.
2019.08.29
  - Fixed a bug where the fpn conv layers weren't getting initialized with xavier since they were being overwritten by jit modules (see #127).
2019.08.04
  - Improved the matching algorithm used to match anchors to gt by making it less greedy (see #104).
2019.06.27
  - Sped up save video by ~8 ms per frame because I forgot to apply a speed fix I applied to the other modes.
```
