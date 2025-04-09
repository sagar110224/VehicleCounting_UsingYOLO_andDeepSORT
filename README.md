# VehicleCounting_UsingYOLO_andDeepSORT
This program is used to count vehicles which crosses the defined line. It counts vehicle in four categories: car,motorbike,bus,truck. If any vehicle is not able to be identified then it is designated as unknown. Bounding box along with unique ID and vehicle class will also be shown while counting

To draw the line, it will open the first frame of the video. Click on the point where you want the line to start. Then, drag it to the end point of the line. Press 'q' to close the window. The vehicle counting will automatically start.

In the program, in __main__ function, there is a dictionary with name 'config'. It has following parameters which we should customize:
1. input_video: to enter the path of video from which vehicle will be counted
2. output_video: enter the path of video where you want your output video will be saved
3. offset: if distance between line and the point is less than offset, then it will be counted
