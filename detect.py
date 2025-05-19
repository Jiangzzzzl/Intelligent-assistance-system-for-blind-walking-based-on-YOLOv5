# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
#
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from aip import AipSpeech
import torch
import platform
import pathlib
import pygame
import os
import time
import requests
import json
from gtts import gTTS
from aip import AipSpeech
import sounddevice as sd
import soundfile as sf
from gpiozero import Button
import subprocess
import smbus
import math
from smtplib import SMTP_SSL
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import socket



BUTTON_PIN = 17
button = Button(BUTTON_PIN)
fs = 8000  # 采样率
duration = 8  # 录音时长（秒）
output_file = "/home/pi/Desktop/recording.wav"  # 输出文件路径

send_usr = ' '  # 发件人邮箱
send_pwd = 'bjwkieetawivdeci' # 授权码，邮箱设置
reverse = ' '  # 接收者邮箱
content = '<p>佩戴人员出现意外:</p>'

html_img = f'<p>{content}<br><img src="cid:image1"></br></p>' # html格式添加图片
email_server = 'smtp.qq.com'
email_title = 'photo'  # 邮件主题



# 百度语音识别参数
APP_ID = '58104572'
API_KEY = 'G5YvtpeY7FQaGoohNXAzw5Rl'
SECRET_KEY = 'xn4b4KZSVmPS2BOZe3Zypg0KdxXpW3S5'

# 百度语音合成参数
SPEECH_APP_ID = '67134301'
SPEECH_API_KEY = 'mE2XnBJS4nXJ0xTffROMl67i'
SPEECH_SECRET_KEY = 'Dl0Zq8TpiiR0vIUplhHdrNjCgLPvACBC'
pygame.mixer.init()

CHAT_API_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=24.c6b17ae3c57e0a928a09ee8cad30f3d4.2592000.1717484933.282335-67113358"

plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
  
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from distance import roadblocks_distance, fence_distance, box_distance, guideboard_distance, trafficlight_distance, stone_distance, tree_distance, chair_distance, dog_distance, cat_distance, people_distance, car_distance, bicycle_distance, plant_distance, rubbishbin_distance, pole_distance, distributorbox_distance, cart_distance, motorcycle_distance, streetlight_distance, brand_distance
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


def check_internet():
    try:
        socket.create_connection(("www.goole.com",80))
        return True
    except OSError:
        pass
    return False



def send_email():

#    cmd = 'fswebcam /dev/video0 ./img.jpg'
#    m = os.popen(cmd)
#    m.close()
#    cv2.error: OpenCV(4.9.0) :-1: error: (-5:Bad argument) in function 'imwrite'
#> Overload resolution failed:
#>  - img is not a numpy array, neither a scalar
#>  - Expected Ptr<cv::UMat> for argument 'img'
#    cv2.imwrire(save_path,im0)
    msg = MIMEMultipart() # 构建主体
    msg['Subject'] = Header(email_title,'utf-8')  # 邮件主题
    msg['From'] = send_usr  # 发件人
    msg['To'] = Header('a','utf-8') # 收件人--这里是昵称
    # msg.attach(MIMEText(content,'html','utf-8'))  # 构建邮件正文,不能多次构造
    f = open('/home/pi/Desktop/yolov5-master/runs/detect/send/send.jpg', 'rb')  #打开图片
    msgimage = MIMEImage(f.read())
    f.close()
    msgimage.add_header('Content-ID', '<image1>')  # 设置图片
    msg.attach(msgimage)
    msg.attach(MIMEText(html_img,'html','utf-8'))  # 添加到邮件正文
    try:
        smtp = SMTP_SSL(email_server)  #指定邮箱服务器
        smtp.connect(email_server, 465)
        smtp.login(send_usr,send_pwd)  # 登录邮箱
        smtp.sendmail(send_usr,reverse,msg.as_string())  # 分别是发件人、收件人、格式
        smtp.quit()  # 结束服务
        print('邮件发送完成--')
    except:
        print('发送失败')




def text_to_speech(text, filename):
    if check_internet():
        client = AipSpeech(SPEECH_APP_ID, SPEECH_API_KEY, SPEECH_SECRET_KEY)

        result = client.synthesis(text, 'zh', 1, {
            'vol': 5,
            'spd': 5,
            'pit': 5,
            'per': 0
        })

        if not isinstance(result, dict):
            with open(filename, 'wb') as f:
                f.write(result)
            print("语音文件已生成：", filename)
            # 播放生成的语音文件
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
        else:
            print("语音合成失败：", result)
    else:
        pygame.mixer.music.load("internetwarn.mp3")
        pygame.mixer.music.play()


def record_audio(filename):
    print("开始录音...")
    # 开始录音
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    # 保存录音
    sf.write(filename, recording, fs)
    print("录音已保存为", filename)

# 语音识别函数
def recognize_audio(file_path):
    if check_internet():
    
        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        
        with open(file_path, 'rb') as f:
            audio_data = f.read()

        result = client.asr(audio_data, 'wav', 8000, {
            'dev_pid': 1536,  # 普通话(支持简单的英文识别) 请根据文档修改为对应的dev_pid
        })

        if 'result' in result:
            print("识别结果：", result['result'][0])
            return result['result'][0]
        else:
            print("识别失败：", result)
            return None
    else:
        pygame.mixer.music.load("internetwarn.mp3")
        pygame.mixer.music.play()

def chat(text):
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.95,
        "top_p": 0.8,
        "penalty_score": 1,
        "disable_search": False,
        "enable_citation": False,
        "response_format": "text"
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(CHAT_API_URL, headers=headers, data=payload)

    # 检查请求是否成功
    if response.status_code == 200:
        # 将响应解析为 JSON 格式
        response_json = response.json()
        # 提取 result 部分
        result = response_json.get("result")
        if result:
            print("回复：", result)
            return result
        else:
            print("未找到回复")
            return None
    else:
        print("请求失败：", response.text)
        return None

def on_button_pressed():
    # 录音
    record_audio(output_file)
    # 语音识别
    text = recognize_audio(output_file)
    if text:
        # 聊天
        reply = chat(text)
        if reply:
            # 语音合成
            text_to_speech(reply, "output.mp3")


power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

def read_byte(adr):
    return bus.read_byte_data(address, adr)

def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr+1)
    val = (high << 8) + low
    return val

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def dist(a,b):
    return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
    radians = math.atan2(x, dist(y,z))
    return -math.degrees(radians)

def get_x_rotation(x,y,z):
    radians = math.atan2(y, dist(x,z))
    return math.degrees(radians)

def trigger_on_rotation(x_rot, y_rot):
    if abs(x_rot) > 65 or abs(y_rot) > 60:
        print("触发函数：角度大于60度")
        send_email()
        

bus = smbus.SMBus(1) # or bus = smbus.SMBus(1) for Revision 2 boards
address = 0x68       # This is the address value read via the i2cdetect command

# Now wake the 6050 up as it starts in sleep mode
bus.write_byte_data(address, power_mgmt_1, 0)

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):


    button.when_pressed = on_button_pressed
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file) or source.endswith('.txt')
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            time.sleep(0.1)
            gyro_xout = read_word_2c(0x43)
            gyro_yout = read_word_2c(0x45)
            gyro_zout = read_word_2c(0x47)

            accel_xout = read_word_2c(0x3b)
            accel_yout = read_word_2c(0x3d)
            accel_zout = read_word_2c(0x3f)

            accel_xout_scaled = accel_xout / 16384.0  # 倍率：±2g
            accel_yout_scaled = accel_yout / 16384.0
            accel_zout_scaled = accel_zout / 16384.0

            x_rotation = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
            y_rotation = get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)

            print("x rotation: ", x_rotation)
            print("y rotation: ", y_rotation)

            # 触发函数：角度大于60时触发
            trigger_on_rotation(x_rotation, y_rotation)
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
#            pic_dir = str(save_dir)+'/pic'
#            if not os.path.exists(pic_dir):
#                os.makedirs(pic_dir)
#            pic_path=pic_dir+'\\'+str(p.stem)+(''if dataset.mode =='image' else f'_{frame}')
            p = Path(p)  # to Path
            save_path = 'runs/detect/send/' # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            cv2.imwrite(save_path+'send.jpg',im0)
#            print(save_path)
#           
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        x1 = int(xyxy[0])  # 获取四个边框坐标
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        h = y2 - y1
                        if names[int(cls)] == "roadblocks":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = roadblocks_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
#                            pic= (int(xyxy[0].item())+int(xyxy[2].item()))/2
#                            if pic != 0:
#                                cv2.imwrite("/1.jpeg"+f'{p.stem}.jpg',im0)
#                            else:
#                                im1=cv2.imread('no.jpg',1)
#                               cv2.imwrite(pic_path+f'{p.stem}.jpg',im1)
                                
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有路障".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有路障"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()

                        if names[int(cls)] == "fence":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = fence_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有栏杆".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有栏杆"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()

                        if names[int(cls)] == "box":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = box_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                                               
                                warnstr = "前方{:.2f}米有盒子".format(dis_m)
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "guideboard":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = guideboard_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有指示牌".format(dis_m)
                                
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "trafficlight":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = trafficlight_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有红绿灯".format(dis_m)
                                
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "stone":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = stone_distance(h)

                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有石头".format(dis_m)
                                
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        # if names[int(cls)] == "tree":
                        #     c = int(cls)  # integer class  整数类 1111111111
                        #     label = None if hide_labels else (
                        #         names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                        #     dis_m = tree_distance(h)  # 调用函数，计算行人实际高度
                        #     label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                        #     txt = '{0}'.format(label)
                        #     annotator.box_label(xyxy, txt, color=colors(c, True))
                        if names[int(cls)] == "chair":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = chair_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有椅子".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有椅子"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "dog":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = dog_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有狗".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有狗"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "cat":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = cat_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有猫".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有猫"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        # if names[int(cls)] == "people":
                        #     c = int(cls)  # integer class  整数类 1111111111
                        #     label = None if hide_labels else (
                        #         names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                        #     dis_m = people_distance(h)  # 调用函数，计算行人实际高度
                        #     label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                        #     txt = '{0}'.format(label)
                        #     annotator.box_label(xyxy, txt, color=colors(c, True))
                        if names[int(cls)] == "car":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = car_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有车".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有车"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "bicycle":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = bicycle_distance(h)
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有自行车".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有自行车"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "plant":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = plant_distance(h)  # 调用函数，计算行人实际高度
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有植物".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有植物"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "rubbishbin":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = rubbishbin_distance(h)  # 调用函数，计算行人实际高度
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有垃圾桶".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有垃圾桶"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "pole":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = pole_distance(h)  # 调用函数，计算行人实际高度
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有杆子".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有杆子"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "distributorbox":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = distributorbox_distance(h)  # 调用函数，计算行人实际高度
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有配电箱".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有配电箱"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "cart":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = cart_distance(h)  # 调用函数，计算行人实际高度
                            label += f'  {dis_m}m'  # 将行人距离显示写在标签后
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有推车".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有推车"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "motorcycle":
                            c = int(cls)  # integer class  整数类 1111111111
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = motorcycle_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有摩托车".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有摩托车"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "streetlight":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            dis_m = streetlight_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有路灯".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有路灯"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()
                        if names[int(cls)] == "brand":
                            c = int(cls)
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 111
                            dis_m = brand_distance(h)
                            label += f'  {dis_m}m'
                            txt = '{0}'.format(label)
                            annotator.box_label(xyxy, txt, color=colors(c, True))
                            if dis_m < 2.0:
                                warnstr = "前方{:.2f}米有警示牌".format(dis_m)
                                #warnstr = "前方" + dis_m + "米有警示牌"
                                filename="output.mp3"
                                text_to_speech(warnstr, filename)
                                pygame.mixer.music.load(filename)
                                pygame.mixer.music.play()

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Webcam", 1280, 720)
                cv2.moveWindow("Webcam", 0, 100)
                cv2.imshow("Webcam", im0)
                cv2.waitKey(1)


            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp2/weights/best.pt", help="model path or triton URL")
    # parser.add_argument("--source", type=str, default='D:/b站下载/Download/5.mp4', help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--source", type=str, default='0',help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/mydata.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
