#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 07:51:18 2017

@author: will
"""
import numpy as np
import tensorflow as tf



def xavier_init(inputShape, outputShape): 
    low = -np.sqrt(6.0/(inputShape + outputShape)) 
    return tf.random_uniform((inputShape, outputShape), 
                             minval=low, maxval=-low, 
                             dtype=tf.float32)

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)
    
class MLP(object):
    
    def __init__(self,Shapes,actFuns):
        self.Shapes = Shapes
        self.actFuns = actFuns
        self.W = [tf.Variable(xavier_init(Shapes[i],Shapes[i+1])) 
                                  for i in range(len(Shapes)-1)]
        self.b = [tf.Variable(tf.zeros([Shapes[i]])) for i in range(1,len(Shapes))]
        self.para = self.W.extend(self.b)
        
    def predict(self,X):
        for w,b,fun in zip(self.W, self.b, self.actFuns):
            X = fun(tf.matmul(X,w)+b)
        return X

class multilayerConv(object):
    
    def __init__(self,Shapes,actFuns,Strides,Padding,forward=True,outputShapes=None):
        # Shapes is a list of shape of the form [filter_height, filter_width, in_channels, out_channels]
        self.forward = forward # conv or conv_transpose
        self.Shapes = Shapes
        self.actFuns = actFuns
        self.Strides = Strides
        self.Padding = Padding 
        self.outputShapes = outputShapes
        self.W = [tf.get_variable("W", shape=Shapes[i],
                       initializer=tf.contrib.layers.xavier_initializer()) 
                                                  for i in range(len(Shapes))]
        self.b = [tf.Variable(tf.zeros([Shapes[i][-1]])) for i in range(1,len(Shapes))]
        self.para = self.W.extend(self.b)
        
    def predict(self,X):
        if self.forward: 
            for w,b,fun,stride,padding in zip(self.W, self.b, self.actFuns,self.Strides,self.Padding):
                X = fun(tf.nn.conv2d(X,w,stride,padding)+b)
        else:
            for w,b,output,fun,stride,padding in zip(self.W, self.b, self.outputShapes, self.actFuns,self.Strides,self.Padding):
                X = fun(tf.nn.conv2d_transpose(X,w,output,stride,padding)+b)            
        return X
    
    
class CGAN(object):
    # Conditional GAN
    
    def __init__(self,GShapes,DShapes,zShape,batchSize,GActFun,DActFun,r,condSize):
        self.X = tf.placeholder(tf.float32,[batchSize,GShapes[-1]])
        self.Z = tf.placeholder(tf.float32,[batchSize,zShape])
        self.CondReal = tf.placeholder(tf.float32,[batchSize,condSize])
        self.CondFake = tf.placeholder(tf.float32,[batchSize,condSize])
        self.zShape = zShape
        self.r = r
        self.batchSize = batchSize
        
        # Generator
        self.G_MLP = MLP(GShapes,GActFun)
        self.X_fake = self.G_MLP.predict(tf.concat([self.Z,self.CondReal],1))
        
        # Discriminator
        self.D_MLP = MLP(DShapes,DActFun)
        D_fake = self.D_MLP.predict(tf.concat([self.X_fake,self.CondReal],1))
        D_real = self.D_MLP.predict(tf.concat([self.X,self.CondReal],1))
        D_mismatch = self.D_MLP.predict(tf.concat([self.X,self.CondFake],1))
        
        # loss
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))
        self.D_loss = -tf.reduce_mean(tf.log(D_real)+0.5*tf.log(1-D_fake)+0.5*tf.log(1-D_mismatch))
        #loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, 
        #                                                                   labels=tf.ones_like(D_real)*0.9))
        #loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
        #                                                                   labels=tf.zeros_like(D_fake)))
        #self.D_loss = loss_real + loss_fake
        #self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))


        # optimizer
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.r).minimize(self.G_loss,var_list=self.G_MLP.para)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.r).minimize(self.D_loss,var_list=self.D_MLP.para)
        
        # session
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _partial_fit(self,X_np,condreal,condfake):
        _,_,loss_G,loss_D = self.sess.run([self.optimizer_G,self.optimizer_D,self.G_loss,self.D_loss],
                                  {self.X:X_np, self.Z:np.random.randn(self.batchSize,self.zShape),\
                                   self.CondReal:condreal,self.CondFake:condfake})
       
        return loss_G,loss_D
    
    def fit(self,X,Y,iterations,CondSampler):
        # CondSampler generate sample from P(cond) that does not equal to Y, with first dimention being batch size
        
        N = X.shape[0]
        n = N/self.batchSize
        for i in range(iterations):
            index_ = np.random.permutation(N)
            cumLoss_G,cumLoss_D = 0,0
            for j in range(n):
                Loss_G,Loss_D = self._partial_fit(X[index_[j*self.batchSize:(j+1)*self.batchSize]],\
                                                 Y[index_[j*self.batchSize:(j+1)*self.batchSize]],\
                                                 CondSampler(Y[index_[j*self.batchSize:(j+1)*self.batchSize]]))
                cumLoss_G += Loss_G
                cumLoss_D += Loss_D
            print "iter: {}, loss_G: {}, loss_D: {}".format(i,cumLoss_G/n, cumLoss_D/n)        
    
    def sample(self,cond):
        return self.sess.run(self.X_fake,{self.Z:np.random.randn(self.batchSize,self.zShape),\
                                         self.CondG:cond})
    
    
def CondSampler(Y):
    sample = np.zeros_like(Y)
    P = 1-Y
    P = P/(Y.shape[1]-1)
    for i in range(Y.shape[0]):
        sample[i]=np.random.multinomial(1,P[i],1)
    return sample    
        
class DCGAN(object):
    
    def __init__(self,G_cnn_Shapes,G_cnn_actFuns,G_cnn_Strides,G_cnn_Padding,G_cnn_outputShapes,
                      D_cnn_Shapes,D_cnn_actFuns,D_cnn_Strides,D_cnn_Padding,D_MLP_Shapes,D_MLP_actFuns,
                      X_H,X_W,X_D,Z_H,Z_W,Z_D,batchSize,r):
        
        self.X_shape = [batchSize,X_H,X_W,X_D]
        self.X = tf.placeholder(tf.float32,self.X_shape)
        self.Z_shape = [batchSize,Z_H,Z_W,Z_D]
        self.Z = tf.placeholder(tf.float32,self.Z_shape)
        self.r = r
        self.batchSize = batchSize
        
        # Generator
        self.G_CNN = multilayerConv(G_cnn_Shapes,G_cnn_actFuns,G_cnn_Strides,G_cnn_Padding,False,G_cnn_outputShapes)
        self.X_fake = self.G_CNN.predict(self.Z)
        
        # Discriminator
        self.D_CNN = multilayerConv(D_cnn_Shapes,D_cnn_actFuns,D_cnn_Strides,D_cnn_Padding)
        self.D_MLP = MLP(D_MLP_Shapes,D_MLP_actFuns)
        D_predict = lambda x: self.D_MLP.predict(self.D_CNN.predict(x))
        D_fake = D_predict(self.X_fake)
        D_real = D_predict(self.X)
        
        # loss
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))
        self.D_loss = -tf.reduce_mean(tf.log(D_real)+tf.log(1-D_fake))
        #loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, 
        #                                                                   labels=tf.ones_like(D_real)*0.9))
        #loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
        #                                                                   labels=tf.zeros_like(D_fake)))
        #self.D_loss = loss_real + loss_fake
        #self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))


        # optimizer
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.r).minimize(self.G_loss,var_list=self.G_CNN.para)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.r).minimize(self.D_loss,var_list=self.D_CNN.para+self.D_MLP.para)
        
        # session
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _partial_fit(self,X_np):
        #_,_,loss_G,loss_D = self.sess.run([self.optimizer_G,self.optimizer_D,self.G_loss,self.D_loss],
        #                          {self.X:X_np, self.Z:np.random.randn(*self.Z_shape)})
        _,loss_G = self.sess.run([self.optimizer_G,self.G_loss],
                                          {self.Z:np.random.randn(*self.Z_shape)})
        _,loss_D = self.sess.run([self.optimizer_D,self.D_loss],
                                          {self.X:X_np, self.Z:np.random.randn(*self.Z_shape)})        
        return loss_G,loss_D
    
    def fit(self,X,iterations):
        N = X.shape[0]
        n = N/self.batchSize
        for i in range(iterations):
            index_ = np.random.permutation(N)
            cumLoss_G,cumLoss_D = 0,0
            for j in range(n):
                Loss_G,Loss_D = self._partial_fit(X[index_[j*self.batchSize:(j+1)*self.batchSize]])
                cumLoss_G += Loss_G
                cumLoss_D += Loss_D
            print "iter: {}, loss_G: {}, loss_D: {}".format(i,cumLoss_G/n, cumLoss_D/n)        
    
    def sample(self):
        return self.sess.run(self.X_fake,{self.Z:np.random.randn(*self.Z_shape)})