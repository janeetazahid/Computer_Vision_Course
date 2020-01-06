**Image Classification with Fashion MNIST dataset**

The purpose of this assigment was to preform image classification on the MNIST dataset, using **Keras** and **CNNs**.
This dataset includes 10 labels of different clothing types with greyscale images. The traning set has 60,000 images and the testing 
set has 10,000. 

The first set was to do the data processing. For example, normalizing the x dataset and applying one-hot encoding to the y dataset. 

Then the model was build. The model consists of the following layers: 
  1. 2D convolutional layer
  2. Pooling Layer
  3. Flatten Layer
  4. Dense Layer #1 (relu activation)
  5. Dense Layer #2 (softmax activation)
  
The model was tranined and then evaluated, using accuracy, precision, recall and f1 score.

**Language:** Python
**Software:** VSCode
**Libraries:** Matplotlib, Keras, Sklearn
