# You Only Look (a couple times)
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```

This README is in progress. Nothing is final. Read at your own risk. The following sections are more for our own bookkeeping than for your eyes.

## Modifications to [SSD](https://www.cs.unc.edu/~wliu/papers/ssd.pdf) so far
 - Use [ResNet101](https://arxiv.org/pdf/1512.03385.pdf) as a backbone
 - Choose anchor boxes that better cover smaller objects
 - Choose prediction layers that make more sense (compared to [SSD 321/513](https://arxiv.org/pdf/1701.06659.pdf))
 - Use Prediction Modules (type c) from [DSSD](https://arxiv.org/pdf/1701.06659.pdf)
 - Use [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf)-style coordinate regressions (which better fits our tiling scheme)
 - *Upsample* images to between 600-1000px while maintaining aspect ratio (no fixed input sizes)
