import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from tensorflow import keras
import tensorflow as tf
import joblib 
from sklearn.preprocessing import MinMaxScaler

style.use("ggplot")

scaler = joblib.load("Storage/Scaler.gz")

close_ind = 3

inv_scaler = MinMaxScaler()
inv_scaler.scale_, inv_scaler.min_ = scaler.scale_[close_ind], scaler.min_[close_ind]


def training(train_data: tuple, val_data: tuple, models: dict, epochs=50, batch_size=32, callbacks=[], plot=True):

    results_dict = {"Losses": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "Val_Losses": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "RMSE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "Val_RMSE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "MAE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "Val_MAE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "MSLE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys())),
                    "Val_MSLE": pd.DataFrame(data=np.zeros(shape=(epochs, len(models))), index=range(epochs), columns=list(models.keys()))}
    
    X_train, y_train = train_data
    X_val, y_val = val_data

    for m in models:
        print(f"Training {m}...")

        model = models[m]

        hist = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0, batch_size=batch_size)

        results_dict["Losses"][m] =  hist.history["loss"]
        results_dict["Val_Losses"][m] =  hist.history["val_loss"]
        results_dict["RMSE"][m] =  hist.history["rmse"]
        results_dict["Val_RMSE"][m] =  hist.history["val_rmse"]
        results_dict["MAE"][m] =  hist.history["mae"]
        results_dict["Val_MAE"][m] =  hist.history["val_mae"]
        results_dict["MSLE"][m] =  hist.history["msle"]
        results_dict["Val_MSLE"][m] =  hist.history["val_msle"]
    
    if plot:
        
        for i in range(2):
            if i == 0:
                prefix = ""
            else:
                prefix = "Val_"
            
            
            fig, axes = plt.subplots(2,2, figsize=(30,20))

            ax11, ax12 = axes[0][0], axes[0][1]
            ax21, ax22 = axes[1][0], axes[1][1]

            ax11.plot(range(int(epochs/5), epochs), results_dict[prefix+"Losses"].iloc[int(epochs/5):], label=list(models.keys()))
            ax11.legend(loc="best")
            ax11.set_title(prefix+"Losses")

            ax12.plot(range(int(epochs/5), epochs), results_dict[prefix+"RMSE"].iloc[int(epochs/5):], label=list(models.keys()))
            ax12.legend(loc="best")
            ax12.set_title(prefix+"RMSE")

            ax21.plot(range(int(epochs/5), epochs), results_dict[prefix+"MAE"].iloc[int(epochs/5):], label=list(models.keys()))
            ax21.legend(loc="best")
            ax21.set_title(prefix+"MAE")

            ax22.plot(range(int(epochs/5), epochs), results_dict[prefix+"MSLE"].iloc[int(epochs/5):], label=list(models.keys()))
            ax22.legend(loc="best")
            ax22.set_title(prefix+"MSLE")

            plt.show()
    
    return results_dict

def evaluation(results_dict):

    columns = list(results_dict.keys())
    index = [i for i in results_dict["Losses"].columns]

    eval_array = np.zeros(shape=(len(index), len(columns)))

    analysis_matrix = pd.DataFrame(data=eval_array, index=index, columns=columns)

    for i in index:
        for j in columns:
            analysis_matrix[j].loc[i] = results_dict[j][i].iloc[-1]
    
    return analysis_matrix
    

def plot_preds(true_vals, preds, title="Close"):
    y_true = true_vals
    y_pred = preds

    
    y_true = inv_scaler.inverse_transform(true_vals)
    y_pred = inv_scaler.inverse_transform(preds)


    plt_df = pd.DataFrame(data=y_true, columns=["Y True"])
    plt_df["Y Pred"] = y_pred

    plt_df.plot(figsize=(12,8))
    plt.title(title)
    plt.show()

    return plt_df




    









                            
    