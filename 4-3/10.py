# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:56:42 2018

@author: kobe24

主要内容:运用yolo算法实现car detection
主要问题:3.2中加载预处理好的yolo.h5文件总是报错(目前还未解决(已解决)),导致后面的步骤无法继续
问题解决:原因分析:照网上的说法,从课程上下载的yolo.h5文件与自己电脑的生成环境不匹配
        解决办法:手动下载训练好的参数建立模型(详见Word文件(步骤))
特别注意:这一节涉及的文件都在'C:/Users/kobe24/yad2k'目录下
未完待续......(完)
"""

#Autonomous driving - Car detection
import argparse
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model
from yolo_utils import read_classes,read_anchors,generate_colors,preprocess_image,draw_boxes,scale_boxes
from yad2k.models.keras_yolo import yolo_head,yolo_boxes_to_corners,preprocess_true_boxes,yolo_loss,yolo_body

#1 - Problem Statement

#2 - YOLO
#2.1 - Model details

#2.2 - Filtering with a threshold on class scores
def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=.6):
    box_scores = box_confidence * box_class_probs
    
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1,keepdims=False)
    
    filtering_mask = box_class_scores >= threshold
    
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores,boxes,classes

#2.3 - Non-max suppression
def iou(box1,box2):
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = (xi2-xi1) * (yi2-yi1)
    
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    max_boxes_tensor = K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold,name=None)
    
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores,boxes,classes

#2.4 Wrapping up the filtering
def yolo_eval(yolo_outputs,image_shape=(720.,1280.),max_boxes=10,score_threshold=.6,iou_threshold=.5):
    box_confidence,box_xy,box_wh,box_class_probs = yolo_outputs
    
    boxes = yolo_boxes_to_corners(box_xy,box_wh)
    
    scores,boxes,classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    
    boxes = scale_boxes(boxes,image_shape)
    
    scores,boxes,classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)
    
    return scores,boxes,classes

#3 - Test YOLO pretrained model on images
sess = K.get_session()

#3.1 - Defining classes, anchors and image shape.
class_names = read_classes('C:/Users/kobe24/yad2k/model_data/coco_classes.txt')
anchors = read_anchors('C:/Users/kobe24/yad2k/model_data/yolo_anchors.txt')
image_shape = (300.,450.)

#3.2 - Loading a pretrained model
yolo_model = load_model('C:/Users/kobe24/yad2k/model_data/yolo.h5')
#yolo_model.summary()

#3.3 - Convert output of the model to usable bounding box tensors
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#3.4 - Filtering boxes
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#3.5 - Run the graph on an image
def predict(sess,image_file):
    image,image_data = preprocess_image("C:/Users/kobe24/yad2k/images/"+image_file,model_image_size=(416,416))
    
    out_scores,out_boxes,out_classes = sess.run([scores,boxes,classes],feed_dict = {yolo_model.input:image_data,K.learning_phase():0})
    
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("C:/Users/kobe24/yad2k/images/out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("C:/Users/kobe24/yad2k/images/out", image_file))
    imshow(output_image)

    return out_scores,out_boxes,out_classes

out_scores, out_boxes, out_classes = predict(sess, "car.jpg") 