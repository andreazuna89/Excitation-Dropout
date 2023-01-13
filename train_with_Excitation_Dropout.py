# Training code applying dropout to the ip1 layer of a CNN-2 architecture

import numpy as np
import sys
import os
caffe_root='../' # Change this path according to the location of your Caffe folder 
sys.path.append(caffe_root +'python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

##============ TRAINING PARAMETERS
test_iter= 52
niter = 100000
iter_EB =0
test_interval = 1000
train_interval=40
base_lr = 0.001
momentum = 0.9
weight_decay = 0.005  
##============

solver=None
solver = caffe.get_solver('./solver.prototxt' )
momentum_hist = {}
for layer in solver.net.params:
		m_w = np.zeros_like(solver.net.params[layer][0].data)
		m_b = np.zeros_like(solver.net.params[layer][1].data)
		momentum_hist[layer] = [m_w, m_b]

##============ EXCITATION DROPOUT PARAMETERS		
C = 0.5 # C = 1-P, where P is the base retaining probability
N = solver.net.blobs['ip1'].data.shape[1] # Layer dimension
##============
	
for i in range(niter):
 
##============ EXCITATION DROPOUT PIPELINE
  solver.net.forward()
  caffe.set_mode_eb_gpu()
  solver.net.blobs['ip3'].diff[...] = 0
  for ff in range(solver.net.blobs['ip3'].diff.shape[0]):
    solver.net.blobs['ip3'].diff[ff,int(solver.net.blobs['label'].data[ff])]=1
  solver.net.backward(start = 'ip3',end = 'ip1')
  mask = (solver.net.blobs['ip1'].diff.copy()) #P_EB probability distribution over layer ip1
  caffe.set_mode_gpu()
  for ff in range(solver.net.blobs['ip1'].data.shape[0]):
    mask[ff,:]=1-np.random.binomial(1,(C*(N-1)*mask[ff,:])/((C*N-1)*mask[ff,:]+1-C)) # Mask generation
    scale=mask[ff,:].mean() # Scale parameter, only used  during training
    solver.net.blobs['ip1'].data[ff,:]=np.multiply(mask[ff,:],solver.net.blobs['ip1'].data[ff,:])/scale # Application of the mask
  solver.net.forward(start='ip2')
  solver.net.backward()
  
##============ UPDATE WEIGHTS
  if i > 25000:
    base_lr = 0.0001
  for layer in solver.net.params:
    if layer == 'ip3':
      lr_w_mult = 10
      lr_b_mult = 20
    else:
      lr_w_mult = 1
      lr_b_mult = 2
    momentum_hist[layer][0] = momentum_hist[layer][0] * momentum + (solver.net.params[layer][0].diff + weight_decay * solver.net.params[layer][0].data) * base_lr * lr_w_mult
    momentum_hist[layer][1] = momentum_hist[layer][1] * momentum + (solver.net.params[layer][1].diff + weight_decay * solver.net.params[layer][1].data) * base_lr * lr_b_mult
    solver.net.params[layer][0].data[...] -= momentum_hist[layer][0]
    solver.net.params[layer][1].data[...] -= momentum_hist[layer][1]
    solver.net.params[layer][0].diff[...] *= 0
    solver.net.params[layer][1].diff[...] *= 0
	
##============ ACCURACY AND LOSS VISUALIZATIONS
  if i % train_interval == 0:
    print 'Iteration', i, 'training loss...', solver.net.blobs['loss'].data, 'accuracy...........', solver.net.blobs['accuracy'].data
  if i % test_interval == 0:
    print 'Iteration', i, 'testing...'
    correct = 0
    for test_it in range(test_iter):
      solver.test_nets[0].share_with(solver.net)
      solver.test_nets[0].forward()
      correct += solver.test_nets[0].blobs['accuracy'].data
    print 'TEST ACCURACY....',correct / test_iter
	
##============ SAVE MODEL
solver.snapshot()
