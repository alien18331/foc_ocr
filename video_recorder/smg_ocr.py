#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:09:19 2022

@author: user
"""

# importing required libraries 
import os
import sys
import cv2 
import time 
import datetime
# import random
import configparser
import matplotlib.pyplot as plt
from threading import Thread # library for implementing multi-threaded processing 

import TIS

# deskewing
import numpy as np
import math
# from deskew import determine_skew # for detect angle
from typing import Tuple, Union

from cv2 import VideoWriter_fourcc

from subprocess import call

# from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler

from ftplib import FTP

# parameters
config = configparser.ConfigParser()
config.read('/home/smg/smg/foc/config.ini')

# general
wind_show = config['general'].getboolean('windows_show')

# camera
cam_serial  = config['camera'].get('serial')
cam_width   = config['camera'].getint('width')
cam_height  = config['camera'].getint('height')
cam_exposureTime = config['camera'].getint('exposureTime')

# video
vdo_fps     = config['video'].getfloat('fps')
vdo_hour    = config['video'].getint('hour')
vdo_minutes = config['video'].getint('minutes')
vdo_record  = config['video'].getint('record')

# ftp
ftp_host    = config['ftp'].get('ftp_host')
ftp_account = config['ftp'].get('ftp_account')
ftp_password = config['ftp'].get('ftp_password')
remotepath  = config['ftp'].get('remotepath')


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

class CustomData:
    ''' Example class for user data passed to the on new image callback function
    '''
    def __init__(self, newImageReceived, image):
        self.newImageReceived = newImageReceived
        self.image = image
        self.busy = False

def on_new_image(tis, userdata):
    '''
    Callback function, which will be called by the TIS class
    :param tis: the camera TIS class, that calls this callback
    :param userdata: This is a class with user data, filled by this call.
    :return:
    '''
    # Avoid being called, while the callback is busy
    if userdata.busy is True:
        return

    userdata.busy = True
    userdata.newImageReceived = True
    userdata.image = tis.Get_image()
    userdata.busy = False

def ftpconnection(host,username, password):
    ftp = FTP()
    ftp.set_debuglevel(2)
    ftp.connect(host, 21)
    ftp.login(username, password)
    
    return ftp
    
def uploadfile(ftp, remotepath, localpath):
    bufsize = 1024
    fp = open(localpath,'rb')
    ftp.storbinary('STOR '+remotepath, fp, bufsize)
    ftp.set_debuglevel(0)
    fp.close()

def video_cam():
    
    global wind_show
    global windos_flg
    global frame  
    global vdo_fps
    
    t = time.localtime()
    video_name = time.strftime("%Y%m%d_%H%M00",t) # 20230119_095500
    
    if(int(video_name[-6:-2])>1200 and (int(video_name[-6:-2])<1215)): #1201-1215
        video_name = video_name[:-6]+"120000"
    elif(int(video_name[-6:-2])>1215 and (int(video_name[-6:-2])<1259)): #1216-1259
        video_name = video_name[:-6]+"121500"
    
    print('video recording, {}'.format(video_name))  
    windos_flg = wind_show
    
    print("frame shape: {}".format(frame.shape))        
    # tmp_dir = os.getcwd()+'tmp_avi/'
    tmp_dir='/home/smg/exec/tmp_avi/'
    # tmp_dir = './tmp_avi/'
    ext_tmp = '.avi'
    ext_video = '.mp4'
    
    avi_tmpDir = tmp_dir + 'tmp' + ext_tmp # './tmp_avi/test.avi' 
    fourcc = VideoWriter_fourcc(*"XVID") # MJPG
    
    print("tmp_video: {}".format(avi_tmpDir))
    videoWriter = cv2.VideoWriter(avi_tmpDir, fourcc, vdo_fps, (640,480),1)
    
    print('video streaming..')  
    try:
        d1 = datetime.datetime.now()  
        num_frames_processed = 0 
        
        while True:        
            # frame to video
            videoWriter.write(frame[:,:,:3])
            num_frames_processed += 1 
            
            # break out
            d2 = datetime.datetime.now()
            if(d2-d1).seconds >= vdo_record: # record time
                # close windows
                cv2.destroyAllWindows()
                windos_flg = False
                
                # fps
                elapsed = (d2-d1).seconds + float(int((d2-d1).microseconds/1000)/1000)
                fps = num_frames_processed/elapsed 
                print("done\nFPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))
                num_frames_processed = 0
                
                print("break out trigger..!")
                break            
        
    finally:                
        # end video
        print("Video writting..!")
        videoWriter.release()
                
        print("Video compressing..")
        time.sleep(5)
        
        # compressing, avi to MP4        
        # video_dir = './tmp_avi/compressed.mp4'
        avi_Dir = tmp_dir + video_name + ext_video 
        command = 'ffmpeg -i %s %s' %(avi_tmpDir, avi_Dir)
        call(command.split())
        
        # print("Compressing done!")
        print("video: {}\n".format(avi_Dir))        
        print("ftp uploading..")
        
        ''' FTP '''        
        print("remote: {}".format(remotepath+video_name + ext_video))
        print("local: {}".format(avi_Dir))
        
        ftp = ftpconnection(ftp_host,ftp_account,ftp_password)
        uploadfile(ftp,remotepath+video_name + ext_video ,avi_Dir)        
        print("ftp upload done..!\n")
        
        # clear tmp_avi file
        filelist = os.listdir(tmp_dir)
        if len(filelist)>0:
            for i in range(len(filelist)):
                if "tmp" in filelist[i]:
                    continue
                if filelist[i][:8] != video_name[:8]:
                    os.remove(tmp_dir+filelist[i])
                    
        print("clear tmpfile done..!\n=================================\n")

        
if (__name__ == "__main__"):
        
    print("Task scheduler..")  
    scheduler = BackgroundScheduler(timezone="Asia/Taipei")
    
    # task
    scheduler.add_job(video_cam, 'cron', day_of_week='0-6', hour=vdo_hour, minute=vdo_minutes)    # 1200
    scheduler.add_job(video_cam, 'cron', day_of_week='0-6', hour=vdo_hour, minute=vdo_minutes+15) #1215
    # scheduler.add_job(video_cam, 'cron', day_of_week='0-6', hour=vdo_hour, minute=vdo_minutes+4)
    
    scheduler.start()
    
    print("camera initial..")    
    
    CD = CustomData(False, None)
    
    Tis = TIS.TIS()
    
    print("camera parameter setup..")
    
    # The following line opens and configures the video capture device.
    # Tis.openDevice("49914013", 640, 480, "30/1",TIS.SinkFormats.BGRA, False)
    Tis.openDevice('16024084', cam_width, cam_height, "30/1",TIS.SinkFormats.BGRA, False)
    
    # The next line is for selecting a device, video format and frame rate.
    # if not Tis.selectDevice():
    #     quit(0)
    
    #Tis.List_Properties()
    Tis.Set_Image_Callback(on_new_image, CD)
    
    Tis.Set_Property("TriggerMode", "On")
    
    Tis.Start_pipeline()
    
    # Remove comment below in oder to get a propety list.
    # Tis.List_Properties()
    
    # In case a color camera is used, the white balance automatic must be
    # disabled, because this does not work good in trigger mode
    try:
        Tis.Set_Property("BalanceWhiteAuto", "Continuous")
        # Tis.Set_Property("BalanceWhiteAuto", "Off")
        # Tis.Set_Property("BalanceWhiteRed", 1.2)
        # Tis.Set_Property("BalanceWhiteGreen", 1.0)
        # Tis.Set_Property("BalanceWhiteBlue", 1.4)
        
    except Exception as error:
        print(error)
    
    try:
        # Query the gain auto and current value :
        print("GainAuto : %s " % Tis.Get_Property("GainAuto"))
        print("Gain : %d" % Tis.Get_Property("Gain"))
    
        # Check, whether gain auto is enabled. If so, disable it.
        if Tis.Get_Property("GainAuto"):
            Tis.Set_Property("GainAuto", "Off")
            print("Gain Auto now : %s " % Tis.Get_Property("GainAuto"))
    
        Tis.Set_Property("Gain", 0)
    
        # Now do the same with exposure. Disable automatic if it was enabled
        # then set an exposure time.
        if Tis.Get_Property("ExposureAuto") :
            Tis.Set_Property("ExposureAuto", "Off")
            print("Exposure Auto now : %s " % Tis.Get_Property("ExposureAuto"))
    
        Tis.Set_Property("ExposureTime", cam_exposureTime)
    
    except Exception as error:
        print(error)
        quit()  
            
    error = 0    
    lastkey = 0
    windos_flg = False
    # cv2.namedWindow('Window',cv2.WINDOW_NORMAL)        
            
    print('Camera ready..\n=================================\n')
    
    try:
                
        while lastkey != 27 and error < 5:        
        
            # time.sleep(1)
            Tis.execute_command("TriggerSoftware") # Send a software trigger
    
            # Wait for a new image. Use 10 tries.
            tries = 10
            while CD.newImageReceived is False and tries > 0:
                time.sleep(0.1)
                tries -= 1
    
            # Check, whether there is a new image and handle it.
            if CD.newImageReceived is True:
                CD.newImageReceived = False
                frame = CD.image
                
                
                if(windos_flg):
                    # num_frames_processed += 1                    
                    cv2.imshow('frame' , frame) 
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):                        
                        break
                
    except KeyboardInterrupt:    
        print("User Terminated..!")        
        
    finally:
        
        # terminate process
        print("TIS Stop the threading..!")
        Tis.Stop_pipeline()  
        
        print("Scheduler Shutdown!")
        scheduler.remove_all_jobs()
        scheduler.shutdown(wait=False)
    
    print("camera initial done!!\n=================================\n")  
    
