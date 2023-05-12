''' simple guide
image = Image.open(image_file)
result = azure_ocr.azure_ocr(image_file)
'''

import os
import sys
import cv2
import math
import numpy as np
import json
import datetime
from datetime import date
import time
from time import gmtime, strftime
import requests
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
import logging
import socket

# ftp
from ftplib import FTP

# mqtt
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish


def rotate(image, angle, center=None, scale=1.0):
    
    (h,w) = image.shape[:2]
    
    if center is None:
        center = (w/2, h/2)
        
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated
    
def azure_ocr(frame):    
    

    # Content type
    content_type = 'application/octet-stream'
    headers = {'content-type': content_type}
    
    # API POST
    with open(frame, "rb") as f:
        data = f.read()
        
    # data = frame.tobytes()
    
    ## API POST
    response = requests.post(url, headers=headers, data=data)
    
    ## Get request ID
    ocr_result_url = response.headers['operation-location']
    
    ## delay 
    ocr_result = requests.get(ocr_result_url, time.sleep(1))
    ocr_result_json = json.loads(ocr_result.text)
    
    # print(ocr_result_json)
    # print(type(ocr_result.text))
    
    # split json result
    result_en = len(ocr_result_json['analyzeResult']['readResults'])
    result_dict = {}
    
    
    if(result_en):
        # print("\nOCR identify..")
        ocr_cnt = len(ocr_result_json['analyzeResult']['readResults'][0]['lines'])
        for idx in range(ocr_cnt):
            tmp_result = ocr_result_json['analyzeResult']['readResults'][0]['lines'][idx]['text']
            result_dict[idx]=tmp_result
            
            # print(tmp_result)

    return result_dict

def isfloat(num):
    try:
        float(num)
        return True

    except ValueError:
        return False
   
def isnumber(num):
    try:
        return num.isdigit()
    
    except ValueError:
        return False

def show_literal(img):
    global digitCnts
    
    fig = plt.figure(figsize = (15,7))
    
    idx = 0
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig1 = plt.subplot(181)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)    
    
    idx = 1
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig2 = plt.subplot(182)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 2
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig3 = plt.subplot(183)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 3
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig4 = plt.subplot(184)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 4
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig5 = plt.subplot(185)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 5
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig6 = plt.subplot(186)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 6
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig7 = plt.subplot(187)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    
    idx = 7
    x = digitCnts[idx][0]
    y = digitCnts[idx][1]
    w = digitCnts[idx][2]
    h = digitCnts[idx][3]    
    fig8 = plt.subplot(188)
    roi = img[y:y+h,x:x+w]
    # plt.axis('off')
    plt.imshow(roi)
    plt.show()

def ftpconnection(host,username,password):
    ftp=FTP()
    ftp.set_debuglevel(2)
    ftp.connect(host, 21)
    ftp.login(username, password)
    
    return ftp

def downloadfile(ftp, remoteDir, localDir):
    bufsize = 1024
    fp = open(localDir, 'wb')
    ftp.retrbinary('RETR '+remoteDir,fp.write,bufsize)
    
    ftp.set_debuglevel(0)
    fp.close()
    
def mqtt_init():
    mqtt_client = mqtt.Client()
    # mqtt_client.username_pw_set()
        
    mqtt_client.connect("10.210.2.157", 7883, 60)
    
    return mqtt_client

def get_logger(name, log_level, log_format):
        
    logger = logging.getLogger(name)
    logfile = logDir+"foc_"+strftime("%Y%m%d")+'.log'
    
    if not logger.handlers:    
        logger.setLevel(level=log_level) # logging.DEBUG
        formatter = logging.Formatter(log_FORMAT,datefmt='%Y%m%d %H:%M:%S')
        
        ch = logging.StreamHandler()
        ch.setLevel(level=log_level)
        ch.setFormatter(formatter)
        
        fh = logging.FileHandler(logfile)
        fh.setLevel(level=log_level)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
    
    print(logger.handlers[1])
    print(os.getcwd())
    
    return logger

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8",80))
    ip = s.getsockname()[0]
    s.close()
    return ip

''' parameters '''
url = 'http://10.11.109.75:5100/vision/v3.2/read/analyze?model-version=2022-04-30'
ftpDir = 'ftp:\\10.10.103.128\\123\\FoscamCamera_E8ABFAA66892\\record\\'
tmpDir = 'D:\\00048766\\FOC\\src\\tmp\\' 
logDir = 'D:\\00048766\\FOC\\src\\log\\' if get_ip()=="10.10.103.128" else './log/' 
plt_show = False
# debug_mode = True if get_ip()=="10.11.141.212" else False
debug_mode = False
ftp_fileNotExists = False

# check dir
os.makedirs(tmpDir, exist_ok=True)  
os.makedirs(logDir, exist_ok=True)

# logger
log_LEVEL = 'INFO'
log_FORMAT = '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s'
log_name = 'foc_logger'

logger = get_logger(log_name,log_LEVEL,log_FORMAT)

log = '============ process start ============'
logger.info(log)

# ocr
DIGITS_LOOKUP = {
    (0,0,0,0,0,0,0): 0,
    (1,1,1,0,1,1,1): 0,
    (0,0,1,0,0,1,0): 1,
    (1,0,1,1,1,0,1): 2,
    (1,0,1,1,0,1,1): 3,
    (0,1,1,1,0,1,0): 4,
    (1,1,0,1,0,1,1): 5,
    (1,1,0,1,1,1,1): 6,
    (1,0,1,0,0,1,0): 7,
    (1,1,1,1,1,1,1): 8,
    (1,1,1,1,0,1,0): 9    
}

# mqtt
mqtt_ip = "10.210.2.157"
mqtt_port = 7883
auth = {'username':"smg", 'password': "arc1"}
topic = 'FOC/SOLAR'

if not(debug_mode):
    mqtt_client = mqtt_init()
log = "mqtt initialing.."
logger.info(log)

# video file
if not(debug_mode):
    # file_time = (datetime.datetime.now()+datetime.timedelta(days=-1)).strftime("%Y%m%d_120000") if int(strftime("%H"))<13 else strftime("%Y%m%d_120000")
    if(int(strftime("%H%M"))<1215): # yesterday
        file_time = (datetime.datetime.now()+datetime.timedelta(days=-1)).strftime("%Y%m%d_120000")
    elif(int(strftime("%H%M"))<1230): # today 1200
        file_time = strftime("%Y%m%d_120000")
    elif(int(strftime("%H%M"))<1259): # today 1230
        file_time = strftime("%Y%m%d_121500")
    else: # 
        file_time = strftime("%Y%m%d_121500")
    
    filename = "{}.mp4".format(file_time)
else:
    filename = '20230131_120000.mp4'

videoFile = tmpDir + filename


log = "video file: {}".format(videoFile)
logger.info(log)

if not(os.path.isfile(videoFile)):
    # print("file({}) not exists, ftp downloading..".format(videoFile))
    log = "video file({}) not exists, ftp downloading..".format(videoFile)
    logger.info(log)    

    ''' FTP '''
    ftp_host = "10.10.103.128"
    ftp_account = "smg"
    ftp_password = "smg"
    ftp_videoDir = '123\\FoscamCamera_E8ABFAA66892\\record\\'
    ftp_flist = []
    
    ftp = ftpconnection(ftp_host,ftp_account,ftp_password)
    ftp.cwd(ftp_videoDir)
    ftp_flist = ftp.nlst()
    
    if not filename in ftp_flist:
        log = "ftp file is not exists: {}".format(filename)
        logger.error(log)
        ftp_fileNotExists = True
        # file_time = (datetime.datetime.now()+datetime.timedelta(days=-1)).strftime("%Y%m%d_121500") if int(strftime("%H"))<13 else strftime("%Y%m%d_121500")
        # filename = "{}.mp4".format(file_time)
        # # filename = "{}{:0>2d}{:0>2d}_121500.mp4".format(date.today().year,date.today().month, date.today().day)
        
        # if not filename in ftp_flist:
        #     ftp_fileNotExists = True
        #     log = "ftp file is not exists: {}".format(filename)
        #     logger.error(log)
    
    if ftp_fileNotExists:
        log = "Program terminated!!"
        logger.error(log)
        sys.exit('terminated!!')
        
    downloadfile(ftp, filename, videoFile)
    
    # print("ftp downloading done..!")  
    log = "ftp downloading done..!"
    logger.info(log)

vidcap = cv2.VideoCapture(videoFile)

''' split frame '''
frame_idx = 0
# print('video frame ..')
log = 'frame spliting..'
logger.info(log)

while(True):
    
    success,frame = vidcap.read()
    if(success):
        if(frame_idx%(120)==0):
            
            # cropped, x: virtical, y: horizen
            region_topx = 137
            region_topy = 13
            region_bottomx = region_topx+125
            region_bottomy = region_topy+610
            Cropped = frame[region_topx:region_bottomx, region_topy:region_bottomy]
    
            a = 80
            Cropped[a:,0:150,0] = Cropped[a,20,0]
            Cropped[a:,0:150,1] = Cropped[a,20,1]
            Cropped[a:,0:150,2] = Cropped[a,20,2]
            
            result = Cropped            
            
            # save as image file
            cv2.imwrite(tmpDir+f'frame_{frame_idx}.jpg', result)
    else:
        # print('video fetch Fail..')
        log = 'video fetch Fail..'
        logger.debug(log)
        break
    
    frame_idx = frame_idx + 1
    
vidcap.release()
log = 'done, video capture release!'
logger.info(log)

''' frame processing '''  
for framename in natsorted(os.listdir(tmpDir)):
    if(framename[-3:]=="jpg"):
        frameFile = tmpDir + framename
        
        if(os.path.getsize(frameFile)>17000 and os.path.getsize(frameFile)<20000):   
            
            img = cv2.imread(frameFile)
            
            # warpAffine
            shift = 8
            p1 = np.float32([[shift,0],[img.shape[1],0],[0,img.shape[0]]])
            p2 = np.float32([[0,0],[img.shape[1]-shift,0],[0,img.shape[0]]])
            M = cv2.getAffineTransform(p1,p2)
            warpAffine = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
                    
            grayscale = cv2.cvtColor(warpAffine, cv2.COLOR_BGR2GRAY)
            
            blur_median = cv2.medianBlur(grayscale,3)
            
            # enhance black
            # enhanced = np.where((blur_median>30) & (blur_median<60), blur_median-25, blur_median)
            
            # INV
            _,th_inv = cv2.threshold(blur_median,100,255,cv2.THRESH_BINARY) # [:,:50]
            # _,inv = cv2.threshold(th_inv, 127,255,cv2.THRESH_BINARY_INV)
            
            # padding
            # pad_val = 0
            # pad_size = 100
            # pad = cv2.copyMakeBorder(inv, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(pad_val,pad_val,pad_val))
            
            
            digits = []
            # digit contours list
            digitCnts = []
            # (st_x, st_y, w, h)
            # idx - small numeric
            roi_s0 = (15,72) #(w,h)
            roi_s1 = (44,72) 
            roi_l0 = (56,113) 
            digitCnts.append((5,6,roi_s0[0],roi_s0[1])) # small 0
            digitCnts.append((31,6,roi_s1[0],roi_s1[1])) # small 1
            digitCnts.append((88,6,roi_s1[0],roi_s1[1])) # s1
            # value - large numeric
            st = 233
            offset = 77
            digitCnts.append((st+(offset*0)+0,6,roi_l0[0],roi_l0[1]-0)) # large 0
            digitCnts.append((st+(offset*1)+2,6,roi_l0[0],roi_l0[1]-2)) # l0
            digitCnts.append((st+(offset*2)+1,6,roi_l0[0],roi_l0[1]-3)) # l0
            digitCnts.append((st+(offset*3)+1,6,roi_l0[0],roi_l0[1]-4)) # l0
            digitCnts.append((st+(offset*4)+1,6,roi_l0[0],roi_l0[1]-5)) # l0
            
            # debug - check numeric location
            if(plt_show):
                show_literal(th_inv)
            
            # loop over each of the digits
            for i in range(len(digitCnts)):
                
                (x,y,w,h) = digitCnts[i]
                
                roi = th_inv[y:y+h, x:x+w]
                
                if(plt_show):
                    plt.imshow(roi)
                    plt.show()
                
                
                (roiH, roiW) = roi.shape
                (bwW,bwH) = (int(roiW*0.7), int(roiH*0.1))
                (bhW,bhH) = (int(roiW*0.15), int(roiH*0.3))
                (dW, dh) = (int(roiW*0.3), int(roiH*0.15))
                (dw, dH) = (int(roiW*0.1), int(roiH*0.1))
                
                offset_H = int(roiH*0.41)
                offset_h = int(roiH*0.48)
                offset_w = int(roiW*0.7)
                    
                # define the set of 7 segments
                if i==0:
                    (bhW,bhH) = (int(roiW*0.5), int(roiH*0.2))
                    (dw, dh) = (int(roiW*0.2), int(roiH*0.2))
                    
                    offset = int(roiH*0.42)
                    
                    segments = [
                        ((0,0),         (roiW, 2)), # top
                        ((0,0),         (roiW, 2)), # top-left
                        ((dw,dh),       (dw+bhW,dh+bhH)), # top-right
                        ((0,0),         (roiW, 2)), # center
                        ((0,0),         (roiW, 2)), # bottom-left
                        ((dw,dh+offset),(dw+bhW,dh+bhH+offset)), # bottom-right
                        ((0,0),         (roiW, 2)) #bottom
                    ]
                elif i<3:
                    (bwW,bwH) = (int(roiW*0.8), int(roiH*0.1))
                    (bhW,bhH) = (int(roiW*0.15), int(roiH*0.3))
                    (dW, dh) = (int(roiW*0.2), int(roiH*0.15))
                    (dw, dH) = (int(roiW*0.1), int(roiH*0.1))
                    
                    offset_H = int(roiH*0.41)
                    offset_h = int(roiH*0.48)
                    offset_w = int(roiW*0.7)
                    
                    segments = [
                        ((dW,2),            (bwW, 2+bwH)), # top
                        ((dw,dH),           (dw+bhW,dH+bhH)), # top-left
                        ((dw+offset_w,dH),  (dw+bhW+offset_w,dH+bhH)), # top-right
                        ((dW,2+offset_H),   (bwW, 2+bwH+offset_H)), # center
                        ((dw,dH+offset_h),         (dw+bhW,dH+bhH+offset_h)), # bottom-left
                        ((dw+offset_w,dH+offset_h),         (dw+bhW+offset_w,dH+bhH+offset_h)), # bottom-right
                        ((dW,2+offset_H*2),   (bwW, 2+bwH+offset_H*2)) #bottom
                    ]
                elif i>=3:
                    offset_H = int(roiH*0.45)
                    
                    segments = [
                        ((dW,2),            (bwW, 2+bwH)), # top
                        ((dw,dH),           (dw+bhW,dH+bhH)), # top-left
                        ((dw+offset_w,dH),  (dw+bhW+offset_w,dH+bhH)), # top-right
                        ((dW,2+offset_H),   (bwW, 2+bwH+offset_H)), # center
                        ((dw,dH+offset_h),         (dw+bhW,dH+bhH+offset_h)), # bottom-left
                        ((dw+offset_w,dH+offset_h),         (dw+bhW+offset_w,dH+bhH+offset_h)), # bottom-right
                        ((dW,2+offset_H*2),   (bwW, 2+bwH+offset_H*2)) #bottom
                    ]
                
                on = [0]*len(segments)
                # print("dig: {}".format(on))
            
                # debug - check segment location
                # for k in range(7):
                #     roi = th_inv[y:y+h,x:x+w].copy()
                #     A = cv2.rectangle(roi, segments[k][0],segments[k][1],(255,255,255), -1) # white
                #     plt.imshow(A, 'gray')
                #     plt.show()
                
                # loop over the segments
                for (n, ((xA, yA), (xB, yB))) in enumerate(segments):
                    segROI = roi[yA:yB, xA:xB]
                    # cnt_W = cv2.countNonZero(segROI)
                    # area = (xB-xA)*(yB-yA)
                    
                    area = segROI.shape[0]*segROI.shape[1]
                    cnt_B = area-cv2.countNonZero(segROI)
                    
                    # if(cnt_W/float(area) <0.5): # on
                    #     on[n] = 1
                    
                    if(cnt_B/float(area) > 0.5): # on
                        on[n] = 1
                
                try:
                    digit = DIGITS_LOOKUP[tuple(on)]
                except:
                    digit = 0
                    
                # print("digit: {}".format(digit))
                digits.append(digit)
                on = [0]*len(segments)
                
            index = int(digits[0]*100+digits[1]*10+digits[2])
            value = int(digits[3]*10000+digits[4]*1000+digits[5]*100+digits[6]*10+digits[7])
            
            # print("filename: {}, digit: {}{}{},{}{}{}{}{}, index: {}, value: {}".format(framename, digits[0],digits[1],digits[2],digits[3],digits[4],digits[5],digits[6],digits[7], index, value))
            log = "filename: {}, digit: {}{}{},{}{}{}{}{}, index: {}, value: {}".format(framename, digits[0],digits[1],digits[2],digits[3],digits[4],digits[5],digits[6],digits[7], index, value)
            logger.info(log)
            
            # ocr
            # result_dic = azure_ocr(frameFile)             
            
            ''' mqtt ''' 
            payload_109 = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "109",                                                                           
                       "value": value/10, # float   
                       "code": "EOK",
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_109 = json.dumps(payload_109)
            
            payload_110 = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "110",                                                                           
                       "value": value/10, # float   
                       "code": "EOK",
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_110 = json.dumps(payload_110)
            
            payload_114 = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "114",                                                                           
                       "value": value/10,
                       "code": "EOK",              
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_114 = json.dumps(payload_114)
            
            payload_118 = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "118",                                                                           
                       "value": value/10, 
                       "code": "EOK",               
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_118 = json.dumps(payload_118)
            
            payload_122 = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "122",                                                                           
                       "value": value/10,  
                       "code": "EOK",              
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_122 = json.dumps(payload_122)
            
            payload_n = {"site": "F12AP34",
                       "location": "SB4F",
                       "sensor": "SOLAR",
                       "tag": "n",                                                                           
                       "value": value,    
                       "code": "EOK",            
                       "time" : strftime("%Y%m%d%H%M%S")}
            json_n = json.dumps(payload_n)
            # payload = {}
            # payload[index] = value
            
            # for single publish
            try:
                print("index: {}".format(index))
                if(index==109):
                    if not(debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_109,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )        
                    log = 'mqtt publish: {}'.format(json_109)
                    logger.info(log)
                    time.sleep(1)
                elif(index==110):
                    if not(debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_110,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )        
                    log = 'mqtt publish: {}'.format(json_110)
                    logger.info(log)
                    time.sleep(1)
                elif(index==114):
                    if not(debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_114,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )        
                    log = 'mqtt publish: {}'.format(json_114)
                    logger.info(log)
                    time.sleep(1)
                elif(index==118):
                    if not(debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_118,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )        
                    log = 'mqtt publish: {}'.format(json_118)
                    logger.info(log)
                    time.sleep(1)
                elif(index==122):
                    if not(debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_122,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )        
                    log = 'mqtt publish: {}'.format(json_122)
                    logger.info(log)
                    time.sleep(1)
                else:
                    if (debug_mode):
                        publish.single(
                            topic = topic,
                            payload = json_n,
                            hostname = mqtt_ip,
                            port = mqtt_port,
                            auth = auth
                        )
                        log = 'mqtt publish: {}'.format(json_n)
                        logger.info(log)
                    
            except Exception as e:
                logger.error(e)
                
            
# print('ocr done! \ntemp folder emptying..')
log = 'ocr done!'
logger.info(log)

log = 'temp folder emptying..'
logger.info(log)
''' reset '''
for f in os.listdir(tmpDir):
    if('.jpg' in f):
        os.remove(tmpDir+f)


# print('============ process done ============')
log = '============ process done ============\n'
logger.info(log)