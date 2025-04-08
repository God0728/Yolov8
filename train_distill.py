import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model_t = YOLO(r'weights/yolov8l.pt')  # 此处填写教师模型的权重文件地址
 
    model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏
 
    model_s = YOLO(r'ultralytics/cfg/models/v8/yolov8.yaml')  # 学生文件的yaml文件 or 权重文件地址
 
    model_s.train(data=r'/root/ultralytics-8.3.27/ultralytics/cfg/datasets/coco_yzj.yaml',  #  将data后面替换你自己的数据集地址
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='/root/ultralytics-8.3.27/runs/train',
                name='yolov8s_distill_0303',
                model_t=model_t.model
                )