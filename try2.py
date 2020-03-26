#!/usr/bin/env python

"""main_feeding.py: Main executable file for auto robotic feeding."""

__author__      = "Edwin Foo"
__copyright__   = "Copyright 2019, NYP"
__credits__ = ["Liang HuiHui", "Edwin Foo", "Wong Yau"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Edwin Foo"
__email__ = "edwin_foo@nyp.edu.sg"
__status__ = "Beta"

import herkulexpy3v1 as herkulex
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
import math
import cv2
from openni import openni2
from openni import _openni2 as c_api
from class4_mouth import astras
#from class_motor import*
from test_motorv3v1 import*
from math import *
from threading import Thread
import sys,termios,tty,os
from select import select

#######################
# wiringPi
#######################
import wiringpi as wpi
import time
 
wpi.wiringPiSetup()
wpi.pinMode(0, 1)
wpi.pinMode(1, 1)
######
# variable decalration
########
home_angle = [0.0, 0.0, 0.0, 0.0]
#cap = cv2.VideoCapture(0)
#prediction ='shape_predictor_68_face_landmarks.dat'
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(prediction)
#complete=0


    ##########
def kbhit():
	''' Returns True if keyboard character was hit, False otherwise.
	'''
#        if os.name == 'nt':
#            return msvcrt.kbhit()

#        else:
	[dr,dw,de] = select([sys.stdin], [], [],0)
	return dr != []

def getch():
    ''' Returns a keyboard character after kbhit() has been called.
        Should not be called in the same program as getarrow().
    '''

    s = ''

#        if os.name == 'nt':
#            return msvcrt.getch().decode('utf-8')

#        else:
    return sys.stdin.read(1)
    ##########
""" setup() """
class Rotate_converter:
    #rotate theta 180 anticlock wise then rotate psi clockwise 90 degrees
    # xcos(theta) + zsin(theta)
    # ycos(psi) + [-sin(psi)][-xsin(theta) + z(cos(theta)]
    # ysin(psi) + [cos(psi)][-xsin(theta) + z(cos(theta)]

    def convert(self,coOrdinateX,coOrdinateY,coOrdinateZ):
        R_co_OrdintateX = -coOrdinateX
        R_co_OrdintateY = coOrdinateZ - 75
        R_co_OrdintateZ = coOrdinateY + 490
        
        return R_co_OrdintateX,R_co_OrdintateY,R_co_OrdintateZ
        
class MyThreadmotor(Thread):
    def __init__(self):

        #Thread.__init__(self)#initialize thread for motor
        
#    def threadstart(self):### motor tread start
        self.servos = servo() # intialize the servo
        self.pos_M = [0]*3

        #print('motor',self.pos_M)

###connect to serial port##
        herkulex.connect("/dev/ttyUSB0",115200)
        for motorID in motor : 
            print(motorID)
            herkulex.clear_error(motorID)
            herkulex.torque_on(motorID)    

        #input("Press anything to continue:")
        print("Cal postion A")
        self.servos.CalPos_A(home_angle)
            #
        old_pos = pos_A
            #print("old_pos",old_pos,"pos_A",pos_A)
            #input()
        pTime = homing_time
        goaltime = int(float(pTime / 11.2))
        print("home angle",home_angle,"pTime",pTime,"goalTime",goaltime)
        if goaltime >=255:
           goaltime = 255
        self.servos.moveMotors(home_angle, goaltime) # move the arm into home position. the angles are in degree.
        #input("Press anything to continue:")
            # read_angle()
        print("Cal postion B")
        self.servos.CalPos_B()  #determine the coordinates of the starting position (wrist coordinates) for the scooping action
            #input()
        print("PosB",pos_B)
        print("Initialisation complete")
        self.servos.moveTo(self.servos.pos_B, scoop_flag)
        #time.sleep(0.5)
        #print("Wait for 0.5s")
        #self.servos.read_angle()
        #input("Press anything to continue:")

#run the motor
    def run(self):
#            self.pos_M = [mouthPos[0],mouthPos[1],mouthPos[2]] #-66, -185, 400.
            #pTime = homing_time
#        while cont.upper() == 'Y':
            #self.servos.scoop()
            #time.sleep(5.0)
            #self.servos.read_angle()
            #self.servos.moveTo(self.pos_M, self.servos.scoop_flag)
            #time.sleep(5.0)
            #self.servos.read_angle()
            self.servos.CalPos_B()  # determine the coordinates of the starting position (wrist coordinates) for the scooping action
            self.servos.moveTo(self.servos.pos_B, self.servos.scoop_flag)
            #time.sleep(5.0)
            #self.servos.read_angle() 

            reach_flag=self.servos.checkReach(self.pos_M, self.servos.arm_attr)
            #print(self.pos_M,self.servos.pos_M)
            if (reach_flag):
                #print("continue again5")
                self.servos.scoop()
                #print("continue again2")
                #print(self.pos_M)
                self.servos.moveTo(self.pos_M,self.servos.scoop_flag)
                time.sleep(5)
                self.servos.CalPos_B()
                self.servos.moveTo(self.servos.pos_B, self.servos.scoop_flag)
            else:
                print("Re-enter new coordinates again")
#               cont=input("Press anything to Y to continue:")
    def pass_value(self,mouthPos):
         self.pos_M = [mouthPos[0],mouthPos[1],mouthPos[2]]
         print('pos m', self.pos_M)

# will stop the motor
    def stop(self):
        print('stop')
        for motorID in motor :
         #print(motorID)
            herkulex.set_led(motorID,herkulex.LED_OFF)
            herkulex.clear_error(motorID)
            herkulex.torque_off(motorID)
        herkulex.close()

            

class camera():
    def __init__(self):
        self.ccor=[-66,-185,400]

        Thread.__init__(self) #initialize thread for 3d camera
        #convert
        self.Rotate_convert = Rotate_converter()
        #self.file1 = open("Dmap.txt","w") 
    def run(self):
        self.cam_astras = astras()
        cap = cv2.VideoCapture(0)
        prediction ='shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(prediction)
        complete=0
        _, frame = cap.read()
        frame2= cv2.resize(frame,(640,400))
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        mar = 10
        faces = detector(gray)
#        s = 0
#        done = False
#        while not done:
#            key = cv2.waitKey(1) & 255
             ## Read keystrokes
#            if key == 27:  # terminate
#                print ("\tESC key detected!")
#                done = True

#            elif key == ord("c"):
#                break

            ###rgb#####
#if mouth detection this (x,y) coordinate will transfer to depth
        try:
           #_, frame = cap.read()
           #frame2= cv2.resize(frame,(240,180))
           #gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
           #mar = 10
           #faces = detector(gray)
           #rgb,mouth1,mouth2 = self.cam_astras.get_rgb()          
           #wpi.digitalWrite(0, 0)
           for face in faces:
                a = face.left()
                b = face.top()
                c = face.right()
                d = face.bottom()
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                landmarks = predictor(gray, face)
                x2 = landmarks.part(63).x
                y2 = landmarks.part(63).y
                x3 = landmarks.part(67).x
                y3 = landmarks.part(67).y
                x4 = landmarks.part(49).x
                y4 = landmarks.part(49).y
                x5 = landmarks.part(65).x
                y5 = landmarks.part(65).y
                
                print('x4',x4,'x5',x5)
                print('y4',y4,'y5',y5)
                #cv2.circle(frame, (x1, y1), 4, (255, 0, 0), -1)
                #cv2.circle(frame, (x2, y2), 4, (255, 0, 0), -1)
                A=dist.euclidean(y2,y3)
                print('distance A',A)
                if A>10:
                    print('mouth open')
                    self.mouth_status=1
                else:
                    print('mouth closed')
                    self.mouth_status=0
                cap.release()
                cv2.destroyAllWindows()
                print('cam closed')
               

############# take lanmark point shape 49 and 65 and dive 2 to find the mouth coordinate #####
           self.x1 =(x4+x5) // 2
           print('x1',self.x1)  
           self.y1 =(y4+y5) // 2
           print('y1',self.y1)
           self.distX =abs(x4-x5)
           print('distX',self.distX)
           

           
           #print("distX",self.distX)
#           self.y1 = (mouth1[1] + mouth2[1]) // 2
           #print('x1,y1',self.x1,self.y1,"mouth1",mouth1,"mouth2",mouth2)
# else mouth not detect use this coordinate to the depth
        except ValueError:
           #rgb,mouth1,mouth2 = self.cam_astras.get_rgb()
           self.x1 = 163
           self.y1 = 209
        dmap, d4d = self.cam_astras.get_depth()
          # Mouth_status = 0
              # DEPTH
        #dmap, d4d = self.cam_astras.get_depth()
        #print('dmap',dmap[self.y1 ,self.x1],dmap[mouth1[1],mouth1[0]],dmap[mouth2[1],mouth2[0]])
        #if damp
        #for data in dmap:
        #  print("dmap",data)
        #print("Dmap dmap[self.x1:-5,self.y1:-5])",d4d[self.x1:self.x1-5,self.y1:self.y1-5])
        #save text file as numpy arrary
        #np.savetxt('dmap.txt', d4d, delimiter=',') 
        
        #time.sleep(2.0)
        #tmp = self.cam_astras.mouth_status
        
        if (self.mouth_status == 1):
            print('ok')
            #self.z1 = dmap[self.x1 ,self.y1] #find the central distance from coordinate camera
            if not ((self.x1) > 640 or (self.y1) > 400):
                print('after checking x1,y1',self.x1,self.y1)
                #dmap, d4d = self.cam_astras.get_depth()
                self.z1 = dmap[self.y1 ,self.x1] #find the central distance from coordinate camera
                if (dmap[self.y1 ,self.x1] == 0 or self.z1>1000) and self.distX >=25 and self.distX<=30:
                  self.z1 = 446         
                elif (dmap[self.y1 ,self.x1] == 0 or self.z1>1000) and self.distX >30 and self.distX<=36:
                   self.z1 = 410
#                else:
#                   self.z1 = 0
                print("z1" , self.z1)
                
                if not ((self.z1 <= 0) or (self.z1 > 500)) :
                    tempcor = self.cam_astras.depth_coordinate(self.x1,self.y1,self.z1) # piexel convert to world coordinate in mm)
   
#                    self.ccor = self.cam_astras.depth_coordinate(self.x1,self.y1,self.z1) # piexel convert to world coordinate in mm)
                 #   //cv2.putText(rgb,'Center pixel is {} mm away'.format(dmap[self.x1, self.y1]),(30,80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    print('x , y ,z , coordinate',self.x1 , self.y1, self.z1, tempcor)# reslut from convert depth xyz pixel to mm
                    self.ccor[0],self.ccor[1],self.ccor[2] = self.Rotate_convert.convert(tempcor[0],tempcor[1],tempcor[2])
                else:
                    self.ccor = [-66,-185,400]    
            else:             
                self.ccor = [-66,-185,400] #
        else:
            self.ccor = [-66,-185,400] # pixel convert to world coordinate in mm)
        

        print("rotated postion", self.ccor, "mouth status %d"%(self.cam_astras.mouth_status))
                ## Distance map
#        cv2.putText(rgb,'Center pixel is {} mm away'.format([0.0.0]),(30,80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#        print('x , y ,z , coordinate',self.x1 , self.y1, self.z1, self.ccor)# reslut from convert depth xyz pixel to mm
               ## Display the stream and windows
        #cv2.namedWindow('rgb', 0) #name the window
        #cv2.resizeWindow('rgb', 160, 120) #reshpe the windown
        #cv2.moveWindow('rgb',160,120)#move windown
  
        #cv2.imshow('rgb', rgb)#show the windown
        #cv2.waitKey(1)
        #self.endtread()

        return(self.ccor)

    def endtread(self): 
    ##image  relise
        cv2.destroyAllWindows()#destory windown
        #self.cam_astras.rgb_stream.stop()#stop 3D rbg_stream
        #self.cam_astras.depth_stream.stop()#stop 3D depth stream
        openni2.unload()
        print ("camera closed")

  
def main():
 try:
    done = False
    smotor = MyThreadmotor()
    wpi.digitalWrite(0, 0)
    wpi.digitalWrite(1, 1)
 #   while not done:
    while True: 
       #while not(kbhit()):
            print("Press Ctrl-C to stop")
            #key = getch()
            ## Read keystrokes
            #if ord(key) == 27:  # terminate
            #    print ("\tESC key detected!")
            #    done = True
            #    break
            d3camera = camera()

            wpi.digitalWrite(0, 1)

            detect_pos = d3camera.run()# initialise the start
            d3camera.endtread()
            #d3camera.join() #join thread
            #loc=[38.0, 356.0, 224.0]
            #print("mouth location: ", detect_pos)
            #loc = detect_pos
#           loc=[0, 500, 460.0]
#           detect_pos = [0,300,300]
#           smotor = MyThreadmotor(d3camera.ccor)
            smotor.pass_value(detect_pos)
            wpi.digitalWrite(1, 1)
            smotor.run()
            wpi.digitalWrite(1, 0)
            '''smotor.pass_value(d3camera.ccor)
            smotor.start()# initialise the motor
            smotor.join()# join thread 
            '''
 except KeyboardInterrupt:
    input("Please get ready to hold motors")
    smotor.stop() #motor will stop enter press
   

if __name__== "__main__":
    main()


