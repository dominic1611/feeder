# feeder
This is a python program that uses a depth camera to detect mouth status and activates the motor of the robotic feederto move to the coordiantes of the mouth . This prgram utilises OpenCV , OpenNI , Dlib ,Python . 
Try2.py is the main program while class4_mouth.py contains the camera's rgb and depth details.

ISSUE 1 : The robot feeder was previously equipped with another camera , and currently equipped with ASTRA EMBEDDED S camera . Hence , there has been some inaccuracy regarding the robot's x y z axis. X axis is accurate but Y and Z axis is inaccurate.The previous camera uses 640x480 resolution while current program uses 640x400.

ISSUE 2 : the camera is often unable to detect object( the person's mouth) which should be within 500mm..But instead it detects objects much futher away instead of the mouth.

