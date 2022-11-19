## Stock Price Forecasting with LSTM Based Models

This is a study on applying LSTM based recurrent neural network architectures for forecasting daily closing prices of the IBM stock and evaluating the results of different architectures. Six different models each making use of LSTM cells but with architectural differences are trained on the same dataset, and their results are compared.

### Data

IBM Stock daily OHLCV data from 2000-01-01 to 18-11-2022 downloaded by yfinance library


### Instructions

- The code for custom layers and functions is in the folder Customs
- The code for data gathering and feature engineering is in Data.ipynb
- The code for normalizing and preprocessing data is in PreProcessing.ipynb
- The code for building and training different architectures; comparing and plotting results can be found in Training_and_Results.ipynb

### Results

> Results of the different models on the training and test data set are given below.

> Performance Metrics(On Scaled Data):

Models |	Training Loss	| Validation Loss |	   Training RMSE     |	Validation RMSE |	  Training MAE      |	Validation MAE	 |   Training MSLE      |	Validation MSLE
-------|------------|------------|-----------|-----------|-----------|-----------|-----------|-----------
LSTM   | 0.000086	|0.000094	 |0.013144	 |0.013726	 |0.009415	 |0.009830	 |0.000085	|0.000077
Stacked LSTM	|0.000108	|0.000113	|0.014709	|0.015004	|0.010491	|0.011111|	0.000108|	0.000093
BiLSTM	|0.000100|	0.000139|	0.014168|	0.016651|	0.010364|	0.012132|	0.000101|	0.000115|
ConvBiLSTM|	0.000127	|0.000168|	0.015918|	0.018310|	0.011505|	0.013576|	0.000125|	0.000137
ConvBiLSTM+T|	0.000097|	0.000096	|0.013955|	0.013857|	0.010373	|0.009762	|0.000098|	0.000079
ConvBiLSTM+T+ATT|	0.000082|	0.000084|	0.012786|	0.012937|	0.009253|	0.009010|	0.000081|	0.000068       
______________
______________                  
> Although the results were relatively close, the best performing model was ConvBiLSTM+T+ATT with the architecture Time Embeddings -> Conv1D -> BiLSTM -> Basic Attention -> Dense                         
> The best model achieved MAE of 1.65 dollars and RMSE of 2.36 dollars on the test data.

### Notes

This is just a basic comparison of different architectures but better results can be achieved by;

1) Gathering more data and engineering better/more features
2) Trying out different forecasting techniques such as predicting off of moving averages and/or daily returns
3) Tuning hyperparemeters(especially learning rate, batch_size, epochs and hidden units)
4) Experimenting with different model architectures, losses, optimizers etc...


### References

[1] Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker, [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321), 2019.

[2] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy, [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174/), 2016.
