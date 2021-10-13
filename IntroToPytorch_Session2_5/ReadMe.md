# Problem Statement
![Problem Statement](./assign-1.png)

- Write a neural network that can;
  - Takes 2 inputs:
    - An image from the MNIST dataset (say 5),
    - A random number between 0 and 9, (say 7)
  - Gives two outputs:
    - The "number" that was represented by the MNIST image (predict 5)
    - The "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

# Solution:
## Data Preparation:
Since we have two input values for the model one is MNIST image and another is a random number so we have to prepare our own dataset which can provide these two values to our model input. Also, our output for the model is again 2, first to provide prediction for MNIST and second to provide prediction for SUM. So, our custom dataset should return three values:
- MNIST images
- random number
- target(This will be concatenation output of image prediction and SUM prediction)

Steps for generating dataset:
- First, get the MNIST image as input to our dataset
- Generate random number of size same as MNIST dataset length.
- Generate one hot encoding for the random number which can sent to the model directly as input.
- Calculate, sum of random number and MNIST target
- Create a separate target tensor where the shape will be (len(MNIST images), 2), So if we sending 100 images for training then it will be (100, 2), the first value will be target for MNIST prediction and the second value will be the target for SUM.

Finally return the values by getitem method as (MNIST Image, random number input, concatenated sum)

## Model 
