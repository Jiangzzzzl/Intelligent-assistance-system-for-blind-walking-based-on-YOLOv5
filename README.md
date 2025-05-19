# Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5

Based on the Raspberry Pi Computer Model 4, it develops functionalities such as obstacle recognition and distance detection, speech recognition and chatting, fall detection alarm, and real-time monitoring via a mobile APP.



## 目录

- [上手指南](##上手指南)
  - [开发前的配置&硬件要求](######开发前的配置&要求)
  - [安装步骤](######安装步骤)
- [训练步骤](##训练步骤)
- [使用说明](##使用说明)
- [各模块测试结果](##各模块测试结果)
- [演示视频](##演示视频)
- [使用到的框架](##使用到的框架)
- [版本控制](##版本控制)
- [作者](##作者)
- [鸣谢](##鸣谢)
  
  

### 上手指南

###### 开发前的配置&硬件要求

1. 树莓派cm4+拓展板/树莓派4b
2. python3.9.2
3. USB摄像头
4. MPU6050
5. 喇叭
6. USB麦克风
7. 杜邦线若干

###### **安装步骤**

1. Clone the repo
   
   ```
   https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5.git
   ```

2. 安装依赖
   
   ```
   pip install -r requirements.txt
   ```

3. 单独安装NumPy 1.26.4
   
   ```
   pip install numpy==1.26.4
   ```
   
   
   
   

### 训练步骤

1.数据集准备

收集项目相关图片，利用labbleme等软件进行标注。

此项目标注类别为常见的21类障碍物，数据集和标签文件对应位置为：

```
├─road
    ├─images
    │  ├─test
    │  └─train
    └─labels
        ├─test
        └─train
```

2.修改配置文件



3.运行 train.py

backbone部分修改为fasternet的代码会在 v2.0 整理完后给出



### 使用说明

1.添加发送邮箱和接收邮箱信息

2.配置好百度语音识别参数和百度语音合成参数

教程可参考[ESP32-S3百度文心一言大模型AI语音聊天助手（支持自定义唤醒词训练）【手把手非常详细】【万字教程】_esp32自定义唤醒词-CSDN博客](https://blog.csdn.net/chg2663776/article/details/142203652)

3.阿里云平台配置（代码整理完成后更新）

4.运行 detect.py



### 各模块测试结果

1.障碍物识别和距离检测
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/detect.png)


2.语音识别和聊天
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/aliyun.jpg)
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/llm.png)

3.摔倒检测报警
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/email.jpg)
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/aliyun_2.jpg)


4.手机APP端实时监测
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/APP_1.jpg)
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/APP_2.jpg)

5.solidworks绘制图
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/3D_1.jpg)
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/3D_2.jpg)

6.最终成品状态
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/show2.jpg)
![image](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/show1.jpg)


## 演示视频
<video src="https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/readme_picture/video.mp4"></video>


### 使用到的框架

- [pytorch](https://pytorch.org/)
- [yolov5](https://jquery.com)
- [fasternet](https://jquery.com)
  
  

### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。



### 作者

Jiangzzzzl

2117154720@qq.com

CSDN:柃茶柒fffffff 



### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE](https://github.com/Jiangzzzzl/Intelligent-assistance-system-for-blind-walking-based-on-YOLOv5/blob/main/LICENSE)



### 鸣谢

- [YOLOv5 🚀 in PyTorch](https://github.com/ultralytics/yolov5)
- [FasterNet](https://github.com/JierunChen/FasterNet)
- [YOLOv5+单目测距（python）_yolov5单目测距-CSDN博客](https://blog.csdn.net/qq_45077760/article/details/130261489)
