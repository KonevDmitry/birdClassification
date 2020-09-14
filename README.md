# CV project. Birds images classification.

This repository builds approaches for bird images classifications using kNN based on autoencoder embeddings, using "100-bird-species" dataset (https://www.kaggle.com/gpiosenka/100-bird-species)

## Dataset batch example

![alt text](./resources/birds.png)

# How to use:
Repository contains birdClassification.ipynb file, which includes all code that we wrote.
All needed libraries will be installed during notebook launch. 

# Algorithm description:
1)  Train image auto-encoder from the dataset.
2)  Pass the data through the encoder part to get dimension-ality reduced data.
3)  Run kNN on embedded data.

## Auto-encoder
The only preprocess which we needed was resizing images to 120x120 pixels.
Our  auto-encoder  consists  of  encoder  and  decoder. Encoder   is   expressed   by   3   convolutional   layers,   3   batchnormalization  layers,  3  ReLU  layers,  and  3  MaxPoll  layers.Decoder  consists of  3  transposed layers,  3  ReLU layers,  and2 convolutional transposed layers.

# kNN
Preprocessed data will be passed to kNN, which is not impplemented yet.
