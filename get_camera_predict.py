import os
import numpy as np
import cv2
from RepsycleDatascience.camera_utils import Camera
from segmentation_eval import Segment

def run():
    # camera = Camera('get_cameras', number=0,
    #                 capture_exposure=6000,
    #                 capture_gain=24)
    camera = Camera('usb', number=0,
                    capture_width=1920,
                    capture_height=1280)

    camera.start()
    prediction = Segment('weights/yolact_base_3196_390000.pth')
    cv2.namedWindow("test")
    while True:
        img = camera.get_image()
        mask_entire, _, _, _ = prediction.predict(img)
        cv2.imshow("test", mask_entire)
        # Quit
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    camera.stop()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()