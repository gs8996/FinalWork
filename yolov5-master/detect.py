import argparse
import time
from pathlib import Path
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QSize, QThread, QWaitCondition, QMutex, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPicture
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(self,save_img=True):
    global run_sign,shot_sign,frames
    source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(self.opt.device)
    # device=torch.device('cuda:0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = '', None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    vid_w = Video()
    for path, img, im0s, vid_cap in dataset:
        if not run_sign:
            return
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # print('append')
                frames.append(im0)
                # print(len(frames))
                if shot_sign:
                    cv2.imwrite(r'C:\Users\26782\Desktop\test.jpg', im0)
                    shot_sign = False
                if video_sign:
                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # vid_writer = cv2.VideoWriter(save_dir+'.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer = Video.video_writer(vid_w, str(save_dir), fourcc, fps, w, h)
                    vid_writer.write(im0)
                # print(1)
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # # # `cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # cv2.imshow('frame', im0)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':

            # else:  # 'video'
            #     # if vid_path != save_path:  # new video
            #     # vid_path = save_path
            #     # if isinstance(vid_writer, cv2.VideoWriter):
            #     #     vid_writer.release()  # release previous video writer


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


class Video:
    wr = []
    cnt = 0

    @staticmethod
    def video_writer(self, save_dir, fourcc, fps, w, h):
        self.cnt += 1
        if self.cnt <= 1:
            writer = cv2.VideoWriter(save_dir + '/video.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.wr.append(writer)
        return self.wr[0]


# 设置多线程读取队列中的图片
class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, *args, **kwargs):
        super(Thread, self).__init__(*args, **kwargs)
        self.is_killed = False
        # self._isPause = False
        # self._value = 0
        # self.cond = QWaitCondition()
        # self.mutex = QMutex()

    def killed(self):
        self.is_killed = True

        # self.cond.wakeAll()

    def run(self):
        # cap = cv2.VideoCapture(0)
        while True:
            if run_sign:
                time.sleep(0.02)
                # if self.is_paused:
                #     if len(frames) >= 1:
                #         frames.pop(0)
                    # frame=cv2.imread(r'C:\Users\26782\Pictures\wallpaper\wallhaven-0q7d7d.jpg')
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # h, w, ch = rgbImage.shape
                    # bytesPerLine = ch * w
                    # convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    # # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    # p = convertToQtFormat.scaled(960, 720, Qt.KeepAspectRatio)
                    # p=QImage(r'C:\Users\26782\Pictures\wallpaper\wallhaven-0q7d7d.jpg')
                    # self.changePixmap.emit(p)
                if len(frames) >= 1:
                    frame = frames.pop(0)
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    # p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    p = convertToQtFormat.scaled(960, 720, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)



class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.run_count = 0

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        check_requirements()
        self.initUI()

    # 视频帧处理函数
    @pyqtSlot(QImage)
    def setImage(self, image):
        if not run_sign:
            self.label.setPixmap(QPixmap(r'C:\Users\26782\Pictures\wallpaper\wallhaven-0q7d7d.jpg'))
        else:
            self.label.setPixmap(QPixmap.fromImage(image))
        
        
    def prepare_run(self):
        while True:
            if run_sign:

                with torch.no_grad():
                    if self.opt.update:  # update all models (to fix SourceChangeWarning)
                        for self.opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                            detect()
                            strip_optimizer(self.opt.weights)
                    else:
                        detect(self)
            else:
                break
            time.sleep(0.5)


    def run_event(self):
        global run_sign
        self.run_count += 1
        # 当点击停止后
        if self.run_count % 2 == 0:
            self.b1.setText('运行')
            run_sign = False
            # self.label.setHidden(True)
            # self.label.deleteLater()
            # time.sleep(0.5)
            # self.label.setPixmap(QPixmap(r'C:\Users\26782\Pictures\wallpaper\wallhaven-0q7d7d.jpg'))
        # 当点击运行后
        else:
            run_sign = True
            self.b1.setText('停止')
            self.prepare_run()
            self.th.start()



    def shot_event(self):
        global shot_sign
        shot_sign = True

    def initUI(self):
        self.setWindowTitle(self.title)

        self.setFixedSize(1200, 1000)
        font = QFont('SimSun', 18, QFont.Bold)

        self.b1 = QPushButton('开始运行')
        self.b1.setFixedSize(150, 60)
        self.b1.setFont(font)
        self.b1.clicked.connect(self.run_event)

        self.b2 = QPushButton('截图')
        self.b2.setFixedSize(150, 60)
        self.b2.setFont(font)
        self.b2.clicked.connect(self.shot_event)

        self.b3 = QPushButton('录制视频')
        self.b3.setFixedSize(150, 60)
        self.b3.setFont(font)
        self.b3.clicked.connect(self.shot_event)

        # 设置水平组件，里面填充按钮
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(300, 840, 400, 100))
        self.Hframe = QtWidgets.QHBoxLayout(self.widget)
        self.Hframe.addWidget(self.b1)
        self.Hframe.addWidget(self.b2)
        self.label = QLabel(self)
        self.label.move(120, 100)
        self.label.resize(960, 720)
        self.label.setPixmap(QPixmap(r'C:\Users\26782\Pictures\wallpaper\wallhaven-0q7d7d.jpg'))
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()



if __name__ == '__main__':
    # 通过列表向UI的多线程后台传递视频帧
    frames = []
    shot_sign = False
    run_sign = False
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
