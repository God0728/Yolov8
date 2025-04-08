import sys
import argparse
import os
sys.path.append('/root/ultralytics-8.3.27/') # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO(opt.cfg) # 可替换为yolo11n.pt
    model.predict(source='/root/autodl-tmp/eval0303/', save=True, imgsz=640, conf=0.5)
    #model.predict('ultralytics/assets/1.png', save=True, imgsz=640, conf=0.5,)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'/root/ultralytics-8.3.27/runs/detect/yolov8s_yzj_150epoch_0228/weights/best.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)