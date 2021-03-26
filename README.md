# Cryptocurrency Price Prediction

## Introduction

This repo makes use of the state-of-art Deep Learning algorithm to predict the price of Bitcoin, which has the potential to generalize to other cryptocurrency. It leverages models such as CNN and RNN implemented by [Keras](https://github.com/keras-team/keras) running on top of [Tensorflow](https://github.com/tensorflow/tensorflow).

Cryptocurrencies, especially Bitcoin, have been one of the top hit in social media and search engines recently. Their high volatility leads to the great potential of high profit if intelligent inventing strategies are taken. It seems that every one in the world suddenly start to talk about Cryptocurrencies. Unfortunately, due to their lack of indexes, Cryptocurrencies are relatively unpredictable compared to traditional financial instruments. This article aims to teach you how to predict the price of these Cryptocurrencies with Deep Learning using Bitcoin as an example so as to provide insight into the future trend of Bitcoin.

## Getting Started

To run this repo, be sure to install the following environment and library:

1. Python 2.7
2. Tensorflow=1.2.0
3. Keras=2.1.1
4. Pandas=0.20.3
5. Numpy=1.13.3
6. h5py=2.7.0
7. sklearn=0.19.1

## Data is collection

Data for prediction can either collected from Kaggle or Poloniex. To make sure coherence, the column names for data collected from Poloniex are changed to match with Kaggle’s.

1. DataCollection.ipynb
2. PastSampler.ipynb

## Data Preparation

Data collected from source needs to be parsed in order to send to the model for prediction. The input size (N) is 256, while the output size (K) is 16. Note that data collected from Poloniex was ticked on a 5 minute basis. This indicates that the input spans across 1280 minutes, while the output covers over 80 minutes.

After creating the PastSampler class, applied it on the collected data. Since the original data ranges from 0 to over 10000, data scaling is needed to allow the neural network to understand the data easier.

## Building Models

### CNN

A 1D Convolutional Neural Network is expected to capture the data locality well with the kernel sliding across the input data. As shown in the following figure.

![Screen Shot 2021-03-26 at 02 29 17](https://user-images.githubusercontent.com/81108192/112568864-12efed00-8ddb-11eb-90cb-a555a739731a.png)

The first model built is Convolutional Neural Network. The following code set the GPU number “1” to be used (since I have 4, you might set it to any GPU you prefer). Since Tensorflow does not seems to do well when running on multiple GPUs, it is wiser to restrict it to run on only 1 GPU. Don’t worry if you do not have a GPU. Simply ignore these lines.

```
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
```

The code for constructing CNN model is very simple. The dropout layer is for preventing overfitting problem. The loss function is defined as Mean Squared Error (MSE), while the optimizer is the state-of-the-art Adam.

```
model = Sequential()
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
model.add(Dropout(0.5))
model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))
model.compile(loss='mse', optimizer='adam')
```

The only thing you need to worry about is the dimension of input and output between each layer. The equation for computing the output of a certain convolutional layer is:

```
Output time step = (Input time step — Kernel size) / Strides + 1
```

At the end of the file, it was added two callback function, CSVLogger, and ModelCheckpoint. The former one helps me to track all the training and validation progress, while the latter one allows me to store the model’s weight for each epoch.

### LSTM

Long Short Term Memory (LSTM) network is a variation of Recurrent Neural Network (RNN). It was invented to solve the vanishing gradient problem created by vanilla RNN. It is claimed that LSTMs are capable of remembering inputs with longer time steps.

![Screen Shot 2021-03-26 at 02 33 18](https://user-images.githubusercontent.com/81108192/112569204-a3c6c880-8ddb-11eb-869a-a521e8017290.png)

LSTM is relatively easier than CNN to implement as you don’t even need to care about the relationship among kernel size, strides, input size and output size. Just make sure the dimension of input and output is defined correctly in the network.

```
model = Sequential()
model.add(LSTM(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
model.add(Dropout(0.8))
model.add(Dense(output_size))
model.add(LeakyReLU())
model.compile(loss='mse', optimizer='adam')
```

### GRU

Gated Recurrent Units (GRU) is another variation of RNN. Its network structure is less sophisticated than LSTM with one reset and forget gate but getting rid of the memory unit. It is claimed that GRU’s performance is on par with LSTM but more efficient. (which is also true, LSTM takes around 45 secs/ epoch, while GRU takes less than 40 secs/ epoch)

![Screen Shot 2021-03-26 at 02 34 59](https://user-images.githubusercontent.com/81108192/112569317-df619280-8ddb-11eb-8e4d-6eb72dfaee0b.png)

Simply replace the second line of building model in LSTM

```
model.add(LSTM(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
```

with

```
model.add(GRU(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
```

## Result Plotting

Since the result plotting is similar for the three model, CNN’s version will be the ony shown. First, we need to reconstruct the model and load the trained_weights into the model.

Then, we need to invert-scaled the predicted data, which ranges from [0,1] because of the MinMaxScaler used previously.

Both Dataframes for the ground true (actual price) and the predicted price of Bitcoin are set up. For visualization purpose, the plotted figure only shows the data from August 2017 thereafter.

Plot the figure with pyplot. Since the predicted price is on a 16 minute basis, not linking all of them up would allow us to view the result easier. As a result, here the predicted data is plotted as red dot, as “ro” in the third line indicates. The blue line in the below graph represents the ground true (actual data), whereas the red dots represent the predicted Bitcoin price.

![Screen Shot 2021-03-26 at 02 38 14](https://user-images.githubusercontent.com/81108192/112569549-526b0900-8ddc-11eb-9ad3-cc071baaefb5.png)

As you can see from the above figure, the prediction closely resemble the actual price of Bitcoin. To select the best model, I decided to test several kinds of configuration of the network, yielding the below table.

![Screen Shot 2021-03-26 at 02 38 36](https://user-images.githubusercontent.com/81108192/112569579-5eef6180-8ddc-11eb-831f-daa68d40c6fb.png)

Each row of the above table is the model that derives the best validation loss from the total 100 training epochs. From the above result, we can observe that LeakyReLU always seems to yield better loss compared to regular ReLU. However, 4-layered CNN with Leaky ReLU as activation function creates a large validation loss, this can due to wrong deployment of model which might require re-validation. CNN model can be trained very fast (2 seconds/ epoch with GPU), with slightly worse performance than LSTM and GRU. The best model seems to be LSTM with tanh and Leaky ReLU as activation function, though 3-layered CNN seems to be better in capturing local temporal dependency of data.

![Screen Shot 2021-03-26 at 02 39 00](https://user-images.githubusercontent.com/81108192/112569615-6d3d7d80-8ddc-11eb-9aa4-9f3964ec5bea.png)

![Screen Shot 2021-03-26 at 02 39 13](https://user-images.githubusercontent.com/81108192/112569631-7595b880-8ddc-11eb-956e-d2aa0846fc41.png)

Although the prediction seems pretty good, there is a concern about overfitting. There is a gap between training and validation loss, (5.97E-06 vs 3.92E-05) when training LSTM with LeakyReLU, regularization should be applied in order to minimize the variance.

## Regularization

To find out the best regularization strategy, several experiments were ran with different L1 and L2 values. First we need to define a new function that facilitate fitting the data into LSTM. Here, use bias regularizer that regularizes over the bias vector as an example.

An experiment is done by repeating training the models for 30 times and each time with 30 epochs.

If you are using Jupyter notebook, you can see the below table directly from the output.

![Screen Shot 2021-03-26 at 02 40 39](https://user-images.githubusercontent.com/81108192/112569751-a83fb100-8ddc-11eb-8564-415baf87bc0a.png)

To visualize the comparison, we can use boxplot:

![Screen Shot 2021-03-26 at 02 41 00](https://user-images.githubusercontent.com/81108192/112569777-b55ca000-8ddc-11eb-89ed-d8f621af4fe8.png)

According to the comparison, it seems that L2 regularizer of coefficient 0.01 on the bias vector derives the best outcome.

To find out the best combination among all the regularizers, including activation, bias, kernel, recurrent matrix, it would be necessary to test all of them one by one, which does not seem practical to my current hardware configuration.

## Conclusion

You have learned:

- How to gather real-time Bitcoin data.
- How to prepare data for training and testing.
- How to predict the price of Bitcoin using Deep Learning.
- How to visualize the prediction result.
- How to apply regularization on the model.

## File Illustration

### There are currently three different models:

1. LSTM.py
2. GRU.py
3. CNN.py (1 dimensional CNN)

### The validation result is plotted in:

1. Plot_LSTM.ipynb
2. Plot_GRU.ipynb
3. Plot_CNN.ipynb

## Run

To run the prediction model, select one of the model. For instance, 
```
python CNN.py
```
To run iPython file, you need to run jupyter notebook
```
jupyter notebook
```
__Be sure to run DataCollection.ipynb and PastSampler.ipynb first to create database for training models.__

## Input & Output & Loss

The input consists of a list of past Bitcoin data with step size of 256.
The output is the predicted value of the future data with step size of 16. Note that since the data is ticked every five minutes, the input data spans over the past 1280 minutes, while the output cover the future 80 minutes. The datas are scaled with MinMaxScaler provided by sklearn over the entire dataset. The loss is defined as Mean Square Error (MSE).

## Result
|Model | #Layers  |  Activation    | Validation Loss   |Test Loss (Scale Inverted) |
|----------| ------------- |------|-------| -----|
|   CNN    | 2       | ReLU       |    0.00029     | 114308 |
|   CNN    | 2       | Leaky ReLU       |    0.00029     | 115525 |
|   CNN    | 3       | ReLU       |    0.00029     | 201718 |
|   CNN    | 3       | Leaky ReLU       |    0.00028     | 108700 |
|   CNN    | 4       | ReLU       |    0.00030     | 117947 |
|   CNN    | 4       | Leaky ReLU       |    0.03217     | 12356304 |
|   LSTM    | 1      | tanh + ReLU       |    0.00007     | 26649 |
|   LSTM    | 1      | tanh + Leaky ReLU       |    0.00004     | 15364 |
|   GRU    | 1      | tanh + ReLU       |    0.00004     | 17667 |
|   GRU    | 1      | tanh + Leaky ReLU       |    0.00004     | 15474 |
|   Baseline (Lag)    | -     | -       |    -     | 19122 |
|   Linear Regression   | -     | -       |    -     | 19789 |



Each row of the above table is the model that derives the best validation loss from the total 100 training epochs. From the above result, we can observe that LeakyReLU always seems to yield better loss compared to regular ReLU. However, 4-layered CNN with Leaky ReLU as activation function creates a large validation loss, this can due to wrong deployment of model which might require re-validation. CNN model can be trained very fast (2 seconds/ epoch with GPU), with slightly worse performance than LSTM and GRU. The best model seems to be LSTM with tanh and Leaky ReLU as activation function, though 3-layered CNN seems to be better in capturing local temporal dependency of data.

## Contributing

Thanks all for your contributions...
    
![Screen Shot 2021-03-21 at 19 11 59](https://user-images.githubusercontent.com/81108192/111917690-519f4380-8a79-11eb-9d01-de457b1655f6.png)
    
ETH WALLET: 0xA1134858c168568CBE37649D16723eC8F782e0A2

![Screen Shot 2021-03-21 at 21 56 54](https://user-images.githubusercontent.com/81108192/111922186-5b807100-8a90-11eb-8504-a3fc3ae35052.png)

BTC WALLET: 3N928MmFq51kbf6fE3fxJbtggBhcjMAhSQ





<div align="center">
	<img src="result/bitcoin2015to2017_close_LSTM_1_tanh_leaky_result.png" width="80%" />
</div>

_LSTM with tanh and Leaky ReLu as activation function._

<div align="center">
	<img src="result/bitcoin2015to2017_close_CNN_3_leaky_result.png" width="80%" />
</div>

_3-layered CNN with Leaky ReLu as activation function._

<div align="center">
	<img src="result/bitcoin2015to2017_close_rw.png" width="80%" />
</div>

_Baseline_

<div align="center">
	<img src="result/bitcoin2015to2017_close_lr.png" width="80%" />
</div>

_Linear Regression_
