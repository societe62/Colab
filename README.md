# 코랩과제

import functools  
import os 

from matplotlib import gridspec 
import matplotlib.pylab as plt    
import numpy as np    
import tensorflow as tf   
import tensorflow_hub as hub    

print("TF Version: ", tf.__version__) 
print("TF-Hub version: ", hub.__version__)  
print("Eager mode enabled: ", tf.executing_eagerly()) 
print("GPU available: ", tf.test.is_gpu_available())  

def crop_center(image): 
  """Returns a cropped square image.""" 
  shape = image.shape 
  new_shape = min(shape[1], shape[2]) 
  offset_y = max(shape[1] - shape[2], 0) // 2 
  offset_x = max(shape[2] - shape[1], 0) // 2 
  image = tf.image.crop_to_bounding_box(  
      image, offset_y, offset_x, new_shape, new_shape)  
  return image  
