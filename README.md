# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```

A simple, fully convolutional model for real-time instance segmentation. This is the code for [our paper](https://arxiv.org/abs/1904.02689), and for the forseeable future is still in development.

Here's a look at our current results for our base model (33 fps on a Titan Xp and 29.8 mAP on COCO's `test-dev`):

![Example 0](data/yolact_example_0.png)

![Example 1](data/yolact_example_1.png)

![Example 2](data/yolact_example_2.png)

# Installation
 - Set up a Python3 environment.
 - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
 - Install some other packages:
   ```Shell
   # Cython needs to be installed before pycocotools
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib 
   ```
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YOLACT on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```


# Evaluation
As of April 5th, 2019 here are our latest models along with their FPS on a Titan Xp and mAP on `test-dev`:

| Image Size | Backbone      | FPS  | mAP  | Weights                                                                                                        |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------|
| 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](http://vision5.idav.ucdavis.edu:6337/weights/yolact_resnet50_54_800000.pth )   |
| 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](http://vision5.idav.ucdavis.edu:6337/weights/yolact_darknet53_54_800000.pth ) |
| 550        | Resnet101-FPN | 33.0 | 29.8 | [yolact_base_54_800000.pth](http://vision5.idav.ucdavis.edu:6337/weights/yolact_base_54_800000.pth )           |
| 700        | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](http://vision5.idav.ucdavis.edu:6337/weights/yolact_im700_54_800000.pth )         |

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands.
## Quantitative Results on COCO
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
# This should get 29.92 validation mask mAP last time I checked.
python eval.py --trained_model=weights/yolact_base_54_800000.pth

# Output a COCOEval json to submit to the website or to use the run_coco_eval.py script.
# This command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

# You can run COCOEval on the files created in the previous command. The performance should match my implementation in eval.py.
python run_coco_eval.py

# To output a coco json file for test-dev, make sure you have test-dev downloaded from above and go
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset
```
## Qualitative Results on COCO
```Shell
# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.3.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --display
```
## Benchmarking on COCO
```Shell
# Run just the raw model on the first 1k images of the validation set
python eval.py --trained_model=weights/yolact_base_54_800000.pth --benchmark --max_images=1000
```
## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time
# I have to work out the kinks for this one. Drawing the frame takes more time than executing the network resulting in sub-30 fps :/
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --video=my_video.mp4

# Process a video and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.3 --top_k=100 --video=input_video.mp4:output_video.mp4
```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python eval.py --help
```


# Training
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   * For Resnet101, download `resnet101_reducedfc.pth` from [here](http://vision5.idav.ucdavis.edu:6337/resnet101_reducedfc.pth).
   * For Resnet50, download `resnet50-19c8e357.pth` from [here](http://vision5.idav.ucdavis.edu:6337/resnet50-19c8e357.pth).
   * For Darknet53, download `darknet53.pth` from [here](http://vision5.idav.ucdavis.edu:6337/darknet53.pth).
 - Run one of the training commands below.
   * Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   * All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5 (suprise). For the 550px models, 1 batch takes up around 1.8 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```
