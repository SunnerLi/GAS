# GAS - Generative Auxiliary Strategy to Accelerate Unconditional GAN Training

[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.7.3-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Python-2.7+-blue.svg)]()

![](https://github.com/SunnerLi/GAS/blob/master/img/structure.jpg)

Motivation
---
How to train generative adversarial network is a very popular issue in recent year. However, you should create the two network to train GAN. The major part of the GAN is generator network that it will learn the join probability of the prior distribution and the observation. On the contrary, the goal of discriminator is just need to give the right guide of JS divergence and lead the generator to revise the parameter toward the right orientation.     
    
However, in order to enlarge the capacity of the discriminator, the size of discriminator is usually big. In some further works (stackGAN), the size of whole network is very large. However, the size of GPU RAM is limited. As the result, **how to shrink the size of discriminator** and raise the utilization of GPU RAM is an important issue.     

Abstraction
---
This work discusses the strategy to reduce the utilization of discriminator, but remain the performance of unconditional training. We discuss for 6 cases and show the result at below. The problem we do the experiment is MNIST LSGAN[1] training. In this work, we consider depthwise separable convolutions[2], inception[3] and dense block[4] idea to revise the discriminator.      

Case Explain
---
index | model description | size (MB)
----- | ----------------- | ---------
 1 | Original LSGAN discriminator | 777 
 2 | fuse the idea of depthwise separable convolutions and inception | 199 
 3 | Only adopt inception idea with the number of base filter is 32 | **489**
 4 | fuse the idea of depthwise separable convolutions, inception and 4 dense block | 393 
 4 | depthwise conv + inception + dense | 393 
 5 | Only adopt inception idea with the number of base filter is 16 | 361 
 6 | Only adopt inception idea with the number of base filter is 8 | 297 


Environment
---
* CPU: Intel(R) Core(TM) i7-4790
* GPU: GeForce® GTX 1070
* RAM: 32GB
* OS : Ubuntu 16.04 

Result
---
![](https://github.com/SunnerLi/GAS/blob/master/img/generator_loss.png)    

In our experiments, we train the models for 2000 iterations, and capture the information in each 200 iterations. First, we make a simple conclusion for GAS idea: **you shouldn't shrink the size of discriminator by yourself!** The above image shows the loss curve of generator. As you can see, if the depth-wise separable convolution idea is adopted, the performance of whole LSGAN will crash! Not only the correct images cannot be rendered, but also the generator can cheat the discriminator easily. The inception-only structure can get great performance, but the size of discriminator didn't reduce very much than the smallest one.    

![](https://github.com/SunnerLi/GAS/blob/master/img/discriminator_loss.png)    

Next, we shows the loss curve of discriminator. Unlike my inference, the loss of discriminator approximate to highest! It means the discriminator cannot judge if the image is fake or not, and the generator cannot learn anything. I might not give any explanation about this result but using original discriminator might be the best idea.    

![](https://github.com/SunnerLi/GAS/blob/master/img/result.gif)    

original | depthwise conv + inception | inception(k=32)
-------- | ---------------------------| -----------------
depthwise conv + inception + dense | inception(k=16) | inception(k=8)
    
At last, we shows the render performance of these six images. As you can see, the performance which model adopt depth-wise separable convolution is very poor! To reduce the size of model, we try to reduce the number of base filter in inception idea. Fortunately, the mode collapse problem occurs. 

Conclusion
---
1. By adopting inception idea with 32 base filter, the whole LSGAN performance is approximate the same as original discriminator, but we reduce about **1.59** time as the origin    
2. The render performance will crash if you adopt depth-wise convolution idea into your discriminator   
3. **It's not recommend to use leaky ReLU in your model, some reason will lead it crash still**     
    

Reference
---
  [1] Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and Stephen Paul Smolley, “Least Squares Generative Adversarial Networks,” arXiv: 1611.04076 [cs.CV], November 2016.    
    
  [2] François Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” arXiv: 1610.02357 [cs.CV], October 2016.    
    
  [3] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich, “Going Deeper with Convolutions,” In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, USA, 8-10, June, 2015, pp. 1-9.    
    
  [4] Gao Huang, Zhuang Liu, Kilian Q. Weinberger, and Laurens van der Maaten, “Densely Connected Convolutional Networks,” arXiv: 1608.06993 [cs.CV], Augest 2016.    
