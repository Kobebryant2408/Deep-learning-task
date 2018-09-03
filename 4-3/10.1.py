# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:10:33 2018

@author: kobe24

对图片做定位时:只需修改3.1中的image_shape与最后一行的image_file('  .jpg')
              前提,图片保存在了相应的目录下;
              图片中有多个目标没有被检测到时,可以改变yolo.eval函数中的参数score_threshold与iou_threshold
特别注意:10.py的所涉及的文件之所以在目录'C:/Users/kobe24/yad2k'下,
        是因为我自己下载训练好的参数构建模型时,git clone 的保存位置在C盘;
        这里,将yad2k文件复制到了E盘,因此,10.1.py的所涉及的文件目录都在
        'E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k'下.

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
class_names = read_classes('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/model_data/coco_classes.txt')
anchors = read_anchors('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/model_data/yolo_anchors.txt')
image_shape = (437.,699.)

#3.2 - Loading a pretrained model
yolo_model = load_model('E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/model_data/yolo.h5')
#yolo_model.summary()

#3.3 - Convert output of the model to usable bounding box tensors
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#3.4 - Filtering boxes
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#3.5 - Run the graph on an image
def predict(sess,image_file):
    image,image_data = preprocess_image("E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/images/"+image_file,model_image_size=(416,416))
    
    out_scores,out_boxes,out_classes = sess.run([scores,boxes,classes],feed_dict = {yolo_model.input:image_data,K.learning_phase():0})
    
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/images/out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("E:/Spyder/神经网络与深度学习/吴恩达 深度学习 编程作业（4-3）(未完成)- Autonomous driving - Car detection/yad2k/images/out", image_file))
    imshow(output_image)

    return out_scores,out_boxes,out_classes

out_scores, out_boxes, out_classes = predict(sess, "cat.jpg") 