# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:29:31 2016

@author: youngwan
"""

import os
import sys
import subprocess
import pandas as pd

#matplotlib.use('Agg')
import matplotlib.pylab as plt

#plt.style.use('ggplot')
def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()  
            if line[0] != '#':
                fields = line.split()
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data


caffe_path = '/home/youngwan/caffe/'

#model_log_path = caffe_path + sys.argv[1]
#learning_curve_path = caffe_path + sys.argv[2]


#model_log_path = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed300x300/KITTI_SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed300x300_2016-11-6-17:41:14.log'
#model_log_path_2 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_300x300_2016-11-14-23:26:29.log'

model_log_path = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed_5th_300x300/KITTI_SSD_Inception_v3_4_preActRes_OriginASP_BN_ImageNet_pretrained_unFreezed_5th_300x300_2016-11-19-15:56:0.log'
model_log_path_2 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_4th300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_No_pretrained_4th300x300_2016-11-17-23:47:35.log'
model_log_path_3 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar_pretrained_3rd_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar_pretrained_3rd_300x300_2016-11-18-10:56:17.log'
model_log_path_4 = caffe_path + 'jobs/New/KITTI/inception_v3/SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar100_pretrained_1st_300x300/KITTI_SSD_Inception_v3_4_preActRes_basic_OriginASP_BN_cifar100_pretrained_1st_300x300_2016-11-23-17:53:56.log'

learning_curve_path = caffe_path + 'jobs/figures/Total_dpi5000.png'


#Get directory where the model logs is saved, and move to it
model_log_dir_path = os.path.dirname(model_log_path)
os.chdir(model_log_dir_path)
command = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
#Read training and test logs
train_log_path = model_log_path + '.train'
test_log_path = model_log_path + '.test'
train_log = pd.read_csv(train_log_path, delim_whitespace=True)
test_log = pd.read_csv(test_log_path, delim_whitespace=True)




#Parsing
#train_data = load_data(train_log_path,1,3)
#test_data = load_data(test_log_path,1,3)


train_loss =  train_log['TrainingLoss']
train_iter = train_log['#Iters']
train_lr = train_log['LearningRate']
test_iter = test_log['#Iters']
test_loss = test_log['TestLoss']
test_acc = test_log['TestAccuracy']
#test_error = 1- test_accuracy


'''
Making learning curve
'''
fig, ax1 = plt.subplots()

#Plotting training and test losses
#train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red',linewidth=2,  alpha=.5)
#test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=2, color='green')


train_loss = plt.plot(train_iter,train_loss, label='Loss', color='red',linewidth=1)


plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.yscale('log')
plt.tick_params(labelsize=10)
plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.)

#Plotting Accuracy
ax2 = plt.twinx()
#test_accuracy, = plt.plot(test_log['#Iters'], test_acc, label='Acc. ImageNet',linewidth=3, color='red', linestyle=':')
train_lr, = plt.plot(test_log['#Iters'], train_lr, label='learning rate',linewidth=3, color='blue', linestyle=':')

ax2.set_ylim(ymin=0.00001, ymax=0.1)
ax2.set_ylabel('Accuracy', fontsize=15)
ax2.tick_params(labelsize=10)
plt.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0.)
plt.title('Training Curve', fontsize=18)

#Saving learning curve
#plt.savefig(learning_curve_path,dpi=2000)
plt.show()


'''
Deleting training and test logs
'''
command = 'rm ' + train_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()

command = command = 'rm ' + test_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()





