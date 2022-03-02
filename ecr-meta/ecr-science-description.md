# Science

This application detects the existence of water (river and other forms of water). This code runs a DeepLab v2 [1] model with a ResNet 101 backbone that is trained with COCO-Stuff [2] dataset. The COCO-Stuff dataset is designed to segment things (person, vehicle, table and others) and stuff (river, window, sea, water and others). The model is trained for both things and stuff, but we are only focusing on to determine if there is water or not. In this situation, the water includes rivers, ponds, and lakes.

# AI at Edge:
This code runs a DeepLab v2 [1] model with a ResNet 101 backbone that is trained with COCO-Stuff [2] dataset. In each run it takes an image from a street view camera and outputs if there is water or not. 

# Using the code
Output: Boolean water existence report (0/1)  
Input: an image (1/30 second)  
Image resolution:  
Inference time:  
Model loading time:  

# Arguments
   '-debug': Debug flag  
   '-stream': ID or name of a stream, e.g. top-camera  
   '-interval': Inference interval in seconds (default = 0, as soon as the plugin can get image)  
   '-sampling-interval': Sampling interval between inferencing (default = -1, no sampling)  
   '-config-path':  Dataset configuration file in YAML  
   '-model-path': PyTorch model to be loaded  
   '-cuda': Use CUDA (default = True)  
   '-crf': Use CRF post-processing (default = True)  

# Ontology:
The code publishes measurements with toptic ‘env.detector.water’. Value for a topic indicates if there is water (1, True) or not (0, False).

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "env.detector.water",
    }
)

# print results in data frame
print(df)
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).


# Reference
[1] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE TPAMI, 2018.  
[2] H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In CVPR, 2018.
