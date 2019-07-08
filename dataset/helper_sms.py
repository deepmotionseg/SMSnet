''' SMSnet:  Semantic  Motion  Segmentationusing  
             Deep  Convolutional  Neural  Networks
 Copyright (C) 2018  Johan Vertens, Abhinav Valada and Wolfram Burgard
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

import tensorflow as tf
import numpy as np
import os
def get_train_batch(config):
    f=os.listdir(config['train_data'])
    img=[]
    label=[]
    flow=[]
    for s in f:
        img.append(os.path.join(config['train_data'],s))
    
    np.random.shuffle(img)
    
    for s in img:
        label.append(s.replace('image_data','label_data'))

    for s in img:
        flow.append(s.replace('image_data','flow_data'))

    dataset = tf.data.Dataset.from_tensor_slices((img, label, flow))
    dataset = dataset.map(lambda x, y, z:parser(x, y, z, config['num_classes']))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.repeat(100000)
    dataset=dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def get_train_data(config):
    iterator = get_train_batch(config)
    dataA,dataB,label= iterator.get_next()
    return [dataA,dataB,label],iterator

def get_test_data(config):
    iterator = get_test_batch(config)
    dataA,dataB,label= iterator.get_next()
    return [dataA,dataB,label],iterator
    
def get_test_batch(config):
    f=os.listdir(config['test_data'])
    img=[]
    label=[]
    flow=[]
    for s in f:
        img.append(os.path.join(config['train_data'],s))
    
    np.random.shuffle(img)
    
    for s in img:
        label.append(s.replace('image_data','label_data'))

    for s in img:
        flow.append(s.replace('image_data','flow_data'))

    dataset = tf.data.Dataset.from_tensor_slices((img, label, flow))
    dataset = dataset.map(lambda x, y, z:parser(x, y, z, config['num_classes']))
    dataset = dataset.batch(config['batch_size'])
    iterator = dataset.make_initializable_iterator()
    return iterator

def compute_output_matrix(label_max, pred_max, output_matrix):
    for i in xrange(output_matrix.shape[0]):
        temp=pred_max==i
        temp_l=label_max==i
        tp=np.logical_and(temp,temp_l)
        temp[temp_l]=True
        fp=np.logical_xor(temp,temp_l)
        temp=pred_max==i
        temp[fp]=False
        fn=np.logical_xor(temp,temp_l)
        output_matrix[i,0]+=np.sum(tp)
        output_matrix[i,1]+=np.sum(fp)
        output_matrix[i,2]+=np.sum(fn)
    return output_matrix

def compute_iou(output_matrix):
    return np.sum(output_matrix[1:,0]/(np.sum(output_matrix[1:,:],1).astype(np.float32)+1e-10))/(output_matrix.shape[0]-1)*100

def parser(i, l, f, n):
    img_contents = tf.read_file(i)
    label_contents=tf.read_file(l)
    flow_contents=tf.read_file(f)
    img = tf.image.decode_png(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    label = tf.image.decode_png(label_contents, channels=1)
    flow = tf.image.decode_png(flow_contents, channels=3)
    flo_r, flo_g, flo_b = tf.split(axis=2, num_or_size_splits=3, value=flow)
    flow = tf.cast(tf.concat(axis=2, values=[flo_b-128, flo_g-128, flo_r]), dtype=tf.float32)
    return img, flow, label


