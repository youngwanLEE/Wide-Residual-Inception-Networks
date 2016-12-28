Code for Wide-Residual-Inception Networks
=============

This code was implemented for Wide-Residual-Inception-Networks(WR-Inception Networks)


This code incorporates materials from:

"Wide Residual Networks", 
http://arxiv.org/abs/1605.07146 (https://github.com/szagoruyko/wide-residual-networks)  
authored by Sergey Zagoruyko and Nikos Komodakis

fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)  
Copyright (c) 2016, Facebook, Inc.  
All rights reserved.




Test error (%, flip/translation augmentation) on CIFAR:  

Network          	 | CIFAR-10 | CIFAR-100 |  
-------------------	 |:--------:|:--------:  
pre-ResNet-164   	 | 5.46     | 24.33  
pre-ResNet-1001  	 | 4.92     | 22.71  
WRN-16-4		 	 | 5.37 	| 24.53  
**WR-Inception**   	 | 5.04 	| 24.16  
**WR-Inception-l2**	 | **4.82** | **23.12**  


<center><img src=https://github.com/youngwanLEE/wide-residual-networks/blob/master/images/Traintime_model_comparison.JPG width=500></center>  
<center><img src=https://github.com/youngwanLEE/wide-residual-networks/blob/master/images/model_test_error_3.png width=500></center>
<center><img src=https://github.com/youngwanLEE/wide-residual-networks/blob/master/logs/wide_inception_v2_8_2004424662/WR-Inception-l2.jpg width=500></center>