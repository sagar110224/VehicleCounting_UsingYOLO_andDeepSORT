from ultralytics import YOLO 
from IPython import display
import cv2
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import math

def get_line_input(action,x,y,flags,*userdata):
    global starting_pt,ending_pt,temp_frame
    if action==cv2.EVENT_LBUTTONDOWN:
        starting_pt=[(x,y)]
    elif action==cv2.EVENT_LBUTTONUP:
        ending_pt=[(x,y)]
    if starting_pt and ending_pt:
        cv2.line(temp_frame,starting_pt[0],ending_pt[0],(0,255,0),2,8)
        cv2.imshow("Window",temp_frame)

def modelandTracker_initialization(pretrained_model,max_age):
    model=YOLO(pretrained_model)
    tracker=DeepSort(max_age=max_age)
    return model,tracker

def initialize_video(config):
    cap=cv2.VideoCapture(config['input_video'])
    return cap

def output_video(config,cap):
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(config['output_video'],fourcc,int(cap.get(5)),(int(cap.get(3)),int(cap.get(4))))
    return out

def detect_vehicles(config,model,box,detected_vehicles,yolo_boxes):
    cls_id=int(box.cls[0])
    label=model.names[cls_id]

    if label in config['vehicle_classes']:
        x1,y1,x2,y2=map(int,box.xyxy[0])
        conf=float(box.conf[0])
        yolo_boxes.append({'label':label,'bbox':[x1,y1,x2,y2]})
        detected_vehicles.append(([x1,y1,x2-x1,y2-y1],conf,label))
    return detected_vehicles,yolo_boxes

def count(config,track_id,cx,cy):
    global vehicle_counter,counter,counted_ids,track_id_to_label,starting_pt,ending_pt
    #line_eq
    m=(ending_pt[0][1]-starting_pt[0][1])/(ending_pt[0][0]-starting_pt[0][1])
    dist=abs(cy-starting_pt[0][1]-m*(cx-starting_pt[0][0]))/math.sqrt(1+m**2)
    if dist<=config['offset']:

        if track_id not in counted_ids:
            cls=track_id_to_label.get(track_id,"unknown")
            counted_ids.add(track_id)
            counter+=1
            vehicle_counter[cls]+=1

def draw_trackfeatures(frame,l,t,r,b,cx,cy,track_id):
    global vehicle_counter,counter,counted_ids,track_id_to_label,starting_pt,ending_pt
    cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
    cv2.putText(frame,f'ID:{track_id} {track_id_to_label[track_id]}',(l,t-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,165,0),1)
    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

#Manual mapping
def box_IOU(boxA,boxB):
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    interarea=max(0,xB-xA)*max(0,yB-yA)

    boxAarea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBarea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    iou=interarea/float(boxAarea+boxBarea-interarea+1e-6)
    return iou

def track_vehicles(config,track,frame,yolo_boxes):
    global vehicle_counter,counter,counted_ids,track_id_to_label,starting_pt,ending_pt
    track_id=track.track_id
    l,t,r,b=map(int,track.to_ltrb())
    w=abs(r-l)
    h=abs(b-t)
    cx,cy=l+w//2,t+h//2
    track_box=[l,t,r,b]

    best_IOU=0
    best_label='unknown'

    for det in yolo_boxes:
        iou=box_IOU(track_box,det['bbox'])
        if iou>best_IOU:
            best_IOU=iou
            best_label=det['label']

    track_id_to_label[track_id]=best_label

    count(config,track_id,cx,cy)
    
    draw_trackfeatures(frame,l,t,r,b,cx,cy,track_id)

def insert_line_and_text(config,frame):
    global vehicle_counter,counter, counted_ids,track_id_to_label,starting_pt,ending_pt
    cv2.line(frame,starting_pt[0],ending_pt[0],(0,0,255),2)
    cv2.putText(frame,f'{config['vehicle_classes'][0]}:{vehicle_counter[config['vehicle_classes'][0]]},{config['vehicle_classes'][1]}:{vehicle_counter[config['vehicle_classes'][1]]},{config['vehicle_classes'][2]}:{vehicle_counter[config['vehicle_classes'][2]]},{config['vehicle_classes'][3]}:{vehicle_counter[config['vehicle_classes'][3]]},unknown:{vehicle_counter['unknown']},Total vehicle count:{counter}',(10,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    cv2.imshow("Window",frame)

def count_vehicles(config,cap,model,tracker,out):
    global vehicle_counter,counter, counted_ids,track_id_to_label,starting_pt,ending_pt
    counted_ids=set()
    vehicle_counter={'unknown':0}
    counter=0
    track_id_to_label={}
    for cls in config['vehicle_classes']:
        vehicle_counter[cls]=0
    
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        results=model(frame,verbose=False)[0]
        detections=[]
        yolo_boxes=[]
        for box in results.boxes:
            detections,yolo_boxes=detect_vehicles(config,model,box,detections,yolo_boxes)
        tracks=tracker.update_tracks(detections,frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_vehicles(config,track,frame,yolo_boxes)
        
        insert_line_and_text(config,frame)

        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

if __name__=='__main__':
    global vehicle_counter,counter, counted_ids,track_id_to_label,starting_pt,ending_pt,temp_frame
    config={
        'input_video':"/Users/drago/Documents/Practicefiles/Data_files/Practice_Project_dataFiles/Vehicle_Counting_data/Mp4_video_files/20231211_VV411_Traffic.mp4",
        'pretrained_model':'yolov8x.pt',
        'vehicle_classes':['car','motorbike','bus','truck'],
        'output_video':'/Users/drago/Documents/Practicefiles/Data_files/Practice_Project_dataFiles/Vehicle_Counting_data/Mp4_video_files/20231211_VV411_Traffic_counted.mp4',
        'DeepSORT_max_age':30,
        'offset':100
    }
    starting_pt=[]
    ending_pt=[]
    #Taking line as input and displaying it
    cap=initialize_video(config)
    ret,frame=cap.read()
    temp_frame=frame.copy()
    cap.release()
    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window",get_line_input)

    k=0
    while k!=113:
        cv2.imshow("Window",temp_frame)
        k=cv2.waitKey(0)
    cv2.destroyAllWindows()

    if starting_pt and ending_pt:
        print(f'Starting {starting_pt}, ending {ending_pt}')
    else:
        print("line not fully defined")
    
    #Initalize model and tracker
    model,tracker=modelandTracker_initialization(config['pretrained_model'],config['DeepSORT_max_age'])

    #Initializing video capturing
    cap=initialize_video(config)

    #Initalizing output video
    out=output_video(config,cap)

    count_vehicles(config,cap,model,tracker,out)

    print(f'{config['vehicle_classes'][0]}:{vehicle_counter[config['vehicle_classes'][0]]},{config['vehicle_classes'][1]}:{vehicle_counter[config['vehicle_classes'][1]]},{config['vehicle_classes'][2]}:{vehicle_counter[config['vehicle_classes'][2]]},{config['vehicle_classes'][3]}:{vehicle_counter[config['vehicle_classes'][3]]},unknown:{vehicle_counter['unknown']},Total vehicle count:{counter}')

    cap.release()
    out.release()
    cv2.destroyAllWindows
    