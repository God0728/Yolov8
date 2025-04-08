import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('/root/ultralytics-8.3.27/ultralytics/cfg/models/v8/yolov8s.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'/root/ultralytics-8.3.27/ultralytics/cfg/datasets/coco.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                #resume='runs/train/exp4/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='/root/ultralytics-8.3.27/runs/train',
                name='yolov8l_0303',
                ) 