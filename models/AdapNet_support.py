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
import network_base

class AdapNet(network_base.Network):
    def __init__(self,num_classes=12,learning_rate=0.001,float_type=tf.float32,weight_decay=0.0005,
                 decay_steps=30000,power=0.9,training=True,ignore_label=True,global_step=0,
                 has_aux_loss=False,weights=None):
        #self.batch=batch
        super(AdapNet, self).__init__()
        self.num_classes=num_classes
        self.learning_rate =learning_rate
        self.weight_decay=weight_decay
        self.initializer='xavier'
        self.has_aux_loss=has_aux_loss
        self.float_type=float_type
        self.power=power
        self.decay_steps=decay_steps
        self.training=training
        self.bn_decay_=0.99
        self.residual_units=[3,4,6,3]
        self.filters=[256,512,1024,2048]
        self.strides=[1,2,2,1]
        self.global_step=global_step
        if self.training:
            self.keep_prob=0.3
        else:
            self.keep_prob=1.0
        if ignore_label and weights==None:
            self.weights=tf.ones(self.num_classes-1)
            self.weights=tf.concat((self.weights,tf.zeros(1)),0)
        elif ignore_label:
            self.weights=weights
        else:
            self.weights=tf.ones(self.num_classes)
        
        
    def _setup(self,data):   
        self.input_shape=data.get_shape()
        self.conv_3x3_out=self.conv_batchN_relu(data,3,1,3,name='conv0')
        self.conv_7x7_out=self.conv_batchN_relu(self.conv_3x3_out,7,2,64,name='conv1')
        self.max_pool_out=self.pool(self.conv_7x7_out,3,2)
        
        ##block1
        self.m_b1_out=self.unit_v1(self.max_pool_out,self.filters[0],1,1,1,shortcut=True)
        for unit_index in range(1,self.residual_units[0]):
            self.m_b1_out=self.unit_v1(self.m_b1_out,self.filters[0],1,1,unit_index+1)
        
        ##block2
        self.m_b2_out=self.unit_v1(self.m_b1_out,self.filters[1],self.strides[1],2,1,shortcut=True)
        for unit_index in range(1,self.residual_units[1]-1):
            self.m_b2_out=self.unit_v1(self.m_b2_out,self.filters[1],1,2,unit_index+1)
        self.m_b2_out=self.unit_v3(self.m_b2_out,self.filters[1],2,self.residual_units[1])
        
            
        ##block3
        self.m_b3_out=self.unit_v1(self.m_b2_out,self.filters[2],self.strides[2],3,1,shortcut=True)
        self.m_b3_out=self.unit_v1(self.m_b3_out,self.filters[2],1,3,2)
        for unit_index in range(2,self.residual_units[2]):
            self.m_b3_out=self.unit_v3(self.m_b3_out,self.filters[2],3,unit_index+1)
        
        ##block4
        self.m_b4_out=self.unit_v4(self.m_b3_out,self.filters[3],4,1,shortcut=True)
        for unit_index in range(1,self.residual_units[3]):
            dropout=False
            if unit_index==2:
               dropout=True
            self.m_b4_out=self.unit_v4(self.m_b4_out,self.filters[3],4,unit_index+1,dropout=dropout)
        
        
        
        
    def _create_loss(self,label):
        self.loss=tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.softmax+1e-10),self.weights),axis=[3]))
   
    def create_optimizer(self):
        self.lr=tf.train.polynomial_decay(self.learning_rate,self.global_step,
                                          self.decay_steps, power=self.power)
        self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)
   
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def build_graph(self,data,label=None):
        self._setup(data)
        
       
           
        
        
    
  
def main():
    a=AdapNetv2(384,768,3,1)
    a.build_graph()
    sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
   

if __name__=='__main__':
    main()

