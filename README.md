Image Segmentation Task:


Through this task I tried to design a convolutional neural network based pipeline for image segmentation.I studied from many resources about deep learning based Image segmentation.Image segmentation is used to detect the pixels of same labels in a image.In traditional approaches people were using Graph-Cut and Clustering based approaches for image segmentation,boundary and edge detection.In this task I used VGG16 pretrained model and keras based convnet for training on my data. following packages are the essentials for running this.

1. Python 3
2. Tensorflow
3. keras
4. OpenCV
clone this repo : git clone https://github.com/pkraison/Task.git
extract and go to Task folder, now download dataset, VGG16 model and weights and put them to Data folder after extraction, be sure that the dataset1 and VGG16 model should be in Data folder. weight should be in Task directory:

Task/$ ls
Data   weights  VGGSegnet.py  LoadBatches.py  train.py  predict.py

Task/Data$ ls
dataset1  vgg16_weights_th_dim_ordering_th_kernels.h5  myTest  predictions


for running this project we need to go to Directory Task and:
(I choose to predict for my custom images with model so choosed Data/mytest. for test set prediction
use Data/dataset1/images_prepped_test as test_images argument.)

Task/$ python predict.py  --save_weights_path=weights/ex1.model  --epoch_number=4  --test_images="Data/myTest/"  --output_path="data/predictions/"  --n_classes=2

for training:
I already trained with sample dataset and saved weights to weights directory:
Task/$ python  train.py  --save_weights_path=weights/ex1  --train_images="data/dataset1/images_prepped_train/"  --train_annotations="data/dataset1/annotations_prepped_train/"  --n_classes=2


#weights are saved in weights directory
###SegNet Description
#The architecture consists of a sequence of non-linear processing layers (encoders) and a corresponding set of decoders followed by a pixelwise classifier. Typically, each encoder consists of one or more convolutional layers with batch normalisation and a ReLU non-linearity, followed by non-overlapping maxpooling and sub-sampling. The sparse encoding due to the pooling process is upsampled in the decoder using the maxpooling indices in the encoding sequence. This has the important advantages of retaining high frequency details in the segmented images and also reducing the total number of trainable parameters in the decoders.(Ref.-http://mi.eng.cam.ac.uk/projects/segnet/)

#1. Uses a novel technique to upsample encoder output which involves storing the max-pooling indices used in pooling layer. This gives reasonably good performance and is space efficient
#2. VGG16 with only forward connections and non trainable layers is used as encoder. This leads to very less parameters.

Dataset and weight link:


Dataset:
#dir:
Data/dataset1
Images Folder - For all the training images
1. Annotations Folder - For the corresponding ground truth segmentation images
2. The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.


Conclusion:
I didn't use GPU and a large datasets so my accuracy rate is not good, but if we'll use more data then results can be good. I'll improve this task after getting large amount of data.
