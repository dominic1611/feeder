#!/usr/bin/env python
from __future__ import division
import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
from openni import openni2
from openni import _openni2 as c_api
import time




MOUTH_AR_THRESH = 0.70
(mStart, mEnd) = (49, 65)

#cap = cv2.VideoCapture(0)
# Function to return some pixel information when the window is clicked
refPt = []
selecting = False


class astras:
    def __init__(self):

######### Register the device##################
         openni2.initialize()
         self.dev = openni2.Device.open_any()
########## create the streams ###########
         #self.rgb_stream = self.dev.create_color_stream()
         self.depth_stream = self.dev.create_depth_stream() 
         #self.color_stream = self.dev.create_color_stream()
         self.dev.set_depth_color_sync_enabled(True)  # synchronize the streams 
         self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
         self.depth_stram = self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=400,fps=30))
#         self.depth_stram = self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480,fps=30))
         #self.color_stream = self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,resolutionX=640,resolutionY=480,fps=30))
######### strat the stream #############
         self.depth_stream.start()
         #self.rgb_stream.start()
         
###########################################
################Mouth###############
         self.predictor = predictor_path = '/home/odroid/feeding20190221_Hui/shape_predictor_68_face_landmarks.dat' #Register land marks
         self.detector = dlib.get_frontal_face_detector()     # detector will get the frontal face 
         self.predictor = dlib.shape_predictor(predictor_path)#Register land marks

         self.mouth_status = 0



#####Close or Open mouth#################
    def mouth_aspect_ratio(self,mouth):   
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
        x2 = landmarks.part(63).x
        y2 = landmarks.part(63).y
        x3 = landmarks.part(67).x
        y3 = landmarks.part(67).y
        A = dist.euclidean(mouth[2],mouth[9]) # 51, 59
        B = dist.euclidean(mouth[4], mouth[7]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates

        C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
	# compute the mouth aspect ratio

        mar = (A + B) / (2.0 * C)
        

	    # return the mouth aspect ratio
        return mar



    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    def get_rgb(self):

        bgr = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(400,640, 3)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rects = self.detector(rgb, 1)

###setup  mouth position ###
        for rect in rects:

            shape = self.predictor(rgb, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            #print('point 49 = ', shape[49])
            #print('point 65 =', shape[65])
            mouthMAR = self.mouth_aspect_ratio(mouth)   
            mar = mouthMAR
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(rgb, [mouthHull], -1, (0, 255, 0), 1)# draw mouth
            cv2.putText(rgb, "MAR: {:.2f}".format(mar), (30, 30),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #if A > MOUTH_AR_THRESH:# if mar > 0.7
                #cv2.putText(rgb, "Mouth is Open!", (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                #self.mouth_status = 1
                 
            #else:
                 #cv2.putText(rgb, "Mouth is close!", (30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                 #self.mouth_status = 0

        try:

            return rgb ,shape[49],shape[65]    # if  shape[49],shape[65] detected it will pass through this coordiante and covert,and give a distance from mouth to cmaera
            #return rgb ,[x2,y2],[x3,y3]

        except:

            return rgb ,[138,201],[147,198]# else will use this coordinate,to measure the distacne from image to camera
    
    def get_depth(self):

#    Returns numpy ndarrays representing the raw and ranged depth images.
#    Outputs:
#        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
#        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
#    Note1:
#        fromstring is faster than asarray or frombuffer
#    Note2:
#        .reshape(120,160) #smaller image for faster response
#                OMAP/ARM default video configuration
#        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
#                Requires .set_video_mode


        dmap = np.fromstring(self.depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(640,400)
        d4d = np.uint8(dmap.astype(float) * 255 / 2 ** 12 - 1) # Correct the range. Depth images are 12bits
        #d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
#        openni1 = openni2.convert_depth_to_world(self.depth_stream,115,207,409)#convert depth piexel to to mm .and change the coordinate 0,0,0 to the central of the image   
 #       print(openni1)

 #       middle_index = (self.depth_stream.read_frame().height + 1) * self.depth_stream.read_frame().width // 2
#        print ("middle = 0x%02x" % (self.depth_stream.read_frame().get_buffer_as_uint16()[middle_index]))
        #print("ts = %s" %  (self.depth_stream.read_frame().timestamp))
        return dmap, d4d    
    
    def get_depth_display(self):

    #    Returns numpy ndarrays representing the raw and ranged depth images.
    #    Outputs:
    #        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
    #        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
    #    Note1:
    #        fromstring is faster than asarray or frombuffer
    #    Note2:
    #        .reshape(120,160) #smaller image for faster response
    #                OMAP/ARM default video configuration
    #        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
    #                Requires .set_video_mode


            dmap = np.fromstring(self.depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(640,400)
            d4d = np.uint8(dmap.astype(float) * 255 / 2 ** 12 - 1) # Correct the range. Depth images are 12bits
            d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
    #        openni1 = openni2.convert_depth_to_world(self.depth_stream,115,207,409)#convert depth piexel to to mm .and change the coordinate 0,0,0 to the central of the image   
     #       print(openni1)

            middle_index = (self.depth_stream.read_frame().height + 1) * self.depth_stream.read_frame().width // 2
    #        print ("middle = 0x%02x" % (self.depth_stream.read_frame().get_buffer_as_uint16()[middle_index]))
            #print("ts = %s" %  (self.depth_stream.read_frame().timestamp))
            return dmap, d4d 
####convert depth to world coordinate pixel to mm
    def depth_coordinate(self, x, y, z):
        
        openni1 = openni2.convert_depth_to_world(self.depth_stream,x,y,z) # covert pixel to world coordinate.
        print("depth_coordinate",openni1)
        
        return openni1
#######################################END##########################################

def main():

    ###########    below this part will move to the main_feeding .py  ###############################
    s = 0
    done = False
    then = time.time() #Time before the operations start

    cam_astras = astras()

    while not done:
        key = cv2.waitKey(1) & 255
        ## Read keystrokes
        if key == 27:  # terminate
            print ("\tESC key detected!")
            done = True

        elif key == ord("c"):
            break
    ###rgb#####

    #if mouth detection this (x,y) coordinate will transfer to depth
        try:
            rgb,mouth1,mouth2 = cam_astras.get_rgb()
            x1 = (mouth1[0] + mouth2[0]) // 2 
            y1 = (mouth1[1] + mouth2[1]) // 2
            print(x1,y1)
    #else mouth not detect use this coordinate to the depth
        except ValueError:
            rgb = cam_astras.get_rgb()
            x1 = 163
            y1 = 209
        # DEPTH
        #print("rgb shape",rgb.shape)
        dmap, d4d = cam_astras.get_depth()
        z1 = dmap[y1,x1] 
        #z1 = dmap[x1 ,y1] #replace this x.y through depth
        #print('dmap shape',dmap.shape)
        canvas = np.hstack((d4d, rgb))
        #coordinate pixel to mm
        ccor = cam_astras.depth_coordinate(x1,y1,z1)
        ## Distance map
     
     #   cv2.putText(rgb,'Center pixel is {} mm away'.format(dmap[x1, y1]),(30,80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)#136.182
        #cv2.putText(rgb,'Center pixel is {} mm away'.format(z1),(30,80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)#136.182
       
        ## Display the stream
        cv2.namedWindow('rgb', 0)
        cv2.resizeWindow('rgb', 1024, 640)
        cv2.moveWindow('rgb',640,320)
        cv2.imshow('rgb', canvas)#rgb)

    now = time.time() #Time after it finished

    print("It took: ", now-then, " seconds")

    cv2.destroyAllWindows()
    cam_astras.rgb_stream.stop()
    cam_astras.depth_stream.stop()
    openni2.unload()
    cap.release()
    print ("END")
    ######################################################
if __name__=="__main__":
    main()

