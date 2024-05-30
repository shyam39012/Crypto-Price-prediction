import pandas as pd
import base64
from math import *
from numpy import concatenate
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error, r2_score,precision_score
from keras.losses import binary_crossentropy
import json
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import numpy as np
import seaborn as sns
from keras.models import load_model
import plotly.graph_objects as go
import streamlit as st



data=pd.read_csv('binancecoin.csv')
data['date'] = pd.to_datetime(data['date'])

df = pd.DataFrame(data)
df.set_index('date', inplace=True)
st.title('Crypto Price Prediction')
option = st.selectbox('select',
    ('Select','Dataset', 'LSTM MODEL', 'GRU MODEL',"RESULT"))

#trainig data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['price'].values.reshape(-1,1))

prediction_days=60

x_train,y_train=[],[]
for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#testing data
test_data=data[len(scaled_data)-560:]
actuals=test_data['price'].values
data['date'] = pd.to_datetime(data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

total_dataset=pd.concat((data['price'],test_data['price']),axis=0)
model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs=model_inputs.reshape(-1,1)
model_inputs=scaler.fit_transform(model_inputs)

x_test=[]
y_test=[]
for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
    y_test.append(model_inputs[x,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#model
lstm_model=load_model('lstm_model.h5')
gru_model=load_model('gru_model.h5')


#lstm prediction
prediction_prices=lstm_model.predict(x_test)
prediction_prices=scaler.inverse_transform(prediction_prices)

lstm_prediction=pd.DataFrame(data={'predictions':prediction_prices.flatten(),'Actuals':actuals.flatten()})
correlation_matrix =lstm_prediction.corr()

#gru_prediction
predictionprices=gru_model.predict(x_test)
predictionprices=scaler.inverse_transform(predictionprices)
gru_prediction=pd.DataFrame(data={'predictions':predictionprices.flatten(),'Actuals':actuals.flatten()})


#RMSE VALUE
lstm_rmse=sqrt(mean_squared_error(actuals,prediction_prices))

gru_rmse=sqrt(mean_squared_error(actuals,predictionprices))


#R2 score

lstm_r2_score=r2_score(actuals,prediction_prices)

gru_r2_score=r2_score(actuals,predictionprices)


def lstm_model_plot(options):

    for i in range(len(options)):
        if(options[i]=='Train and Test'):
            fig=plt.figure(figsize=(16, 6))
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price', fontsize=13)

            # Plotting train and test data
            plt.plot(data['date'], data['price'], linewidth=3, label='Train')
            plt.plot(test_data['date'], test_data['price'], linewidth=3, label='Test')

            plt.legend()

            # Format x-axis as years
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


            plt.tight_layout()
            plt.show()
            st.pyplot(fig)
            st.write('This graph displays the time and date of the Binance Coin training and testing datasets.')

        elif(options[i]=='MSE and MAE'):
            st.write()
            

            # Load the training history from the saved file
            with open('training_history.json', 'r') as f:
                history_data = json.load(f)

            # Plotting loss
            fig=plt.figure(figsize=(16, 6))
            plt.title('Graph between Mean Absolute and Mean Squared Error')
            plt.plot(history_data['loss'], label='MSE')
            plt.plot(history_data['val_loss'], label='MAE')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            st.pyplot(fig)
            

        elif(options[i]=='Actual and Predicted'):
            st.write()
            fig=plt.figure(figsize=(15,6))
            plt.plot(actuals,color='black',label='Actual Prices')
            plt.plot(prediction_prices,color='green',label='Predicted Prices')
            plt.title('Binance price prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            st.pyplot(fig)

        elif(options[i]=='Heat Map'):
            st.write('hello')
            # Create a heatmap to visualize the correlation
            fig=plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap between Actual and Predicted Prices')
            st.pyplot(fig)
        elif(options[i]=='Future Prediction'):
            # Function to make predictions using the LSTM model
            def make_lstm_predictions(model, input_data, scaler):
                # Reshape input data for compatibility with the LSTM model
                #input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
                
                # Make predictions
                prediction = model.predict(input_data)
                
                # Inverse transform the prediction to the original scale
                prediction = scaler.inverse_transform(prediction)
                
                return prediction[0, 0]


            

            n_days = 50  # Adjust as needed
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, n_days + 1)]

            # Make predictions for future dates
            future_predictions = []

            for date in future_dates:
                # Extract the input data for the current date
                # Assuming input window size is 60, adjust as neede

                # Scale the input data
                input_data_scaled =scaler.fit_transform(df['price'].values[-60:].reshape(-1, 1))
                
                # Make predictions using the LSTM model
                prediction = make_lstm_predictions(lstm_model, input_data_scaled, scaler)
                
                # Append the prediction to the list
                future_predictions.append(prediction)
                
                # Update the dataframe with the actual value (for illustration purposes)
                df.loc[date] = prediction

            # Plotting
            fig=plt.figure(figsize=(10, 6))
            plt.plot(data['date'], data['price'], label='Actual Data')
            plt.plot(future_dates, future_predictions, label='Future Predictions')

            plt.title('Time Series Prediction with LSTM')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(fig)


        else:
            st.write('RMSE VALUE OF LSTM',lstm_rmse)
            
            st.write('R2 score of LSTM',lstm_r2_score)

       



def gru_model_plot(options):
    for i in range(len(options)):
        if(options[i]=='Train and Test'):
            fig=plt.figure(figsize=(16, 6))
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price', fontsize=13)

            # Plotting train and test data
            plt.plot(data['date'], data['price'], linewidth=3, label='Train')
            plt.plot(test_data['date'], test_data['price'], linewidth=3, label='Test')

            plt.legend()

            # Format x-axis as years
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


            plt.tight_layout()
            plt.show()
            st.pyplot(fig)
            st.write('This graph displays the time and date of the Binance Coin training and testing datasets.')

        elif(options[i]=='MSE and MAE'):
            st.write()
             
            # Load the training history from the saved file
            with open('training_history_gru.json', 'r') as f:
                history_data = json.load(f)

            # Plotting loss
            fig=plt.figure(figsize=(16, 6))
            plt.title('Graph between Mean Absolute and Mean Squared Error')
            plt.plot(history_data['loss'], label='MSE')
            plt.plot(history_data['val_loss'], label='MAE')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            st.pyplot(fig)
        elif(options[i]=='Actual and Predicted'):
            st.write()
            fig=plt.figure(figsize=(15,6))
            plt.plot(actuals,color='black',label='Actual Prices')
            plt.plot(predictionprices,color='orange',label='Predicted Prices')
            plt.title('Binance price prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            st.pyplot(fig)
        elif(options[i]=='Heat Map'):
            st.write("Correlation Heatmap between Actual and Predicted Prices")

            # Calculate correlation matrix
            correlation_matrix = gru_prediction.corr()

            # Create a heatmap to visualize the correlation
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap between Actual and Predicted Prices')

            # Display the heatmap using Streamlit
            st.pyplot(fig)
        elif(options[i]=='Future Prediction'):
            # Function to make predictions using the LSTM model
            def make_lstm_predictions(model, input_data, scaler):
                # Reshape input data for compatibility with the LSTM model
                #input_data = np.reshape(input_data, (1, input_data.shape[0], 1))
                
                # Make predictions
                prediction = model.predict(input_data)
                
                # Inverse transform the prediction to the original scale
                prediction = scaler.inverse_transform(prediction)
                
                return prediction[0, 0]


            

            n_days = 50  # Adjust as needed
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, n_days + 1)]

            # Make predictions for future dates
            future_predictions = []

            for date in future_dates:
                # Extract the input data for the current date
                # Assuming input window size is 60, adjust as neede

                # Scale the input data
                input_data_scaled =scaler.fit_transform(df['price'].values[-60:].reshape(-1, 1))
                
                # Make predictions using the LSTM model
                prediction = make_lstm_predictions(lstm_model, input_data_scaled, scaler)
                
                # Append the prediction to the list
                future_predictions.append(prediction)
                
                # Update the dataframe with the actual value (for illustration purposes)
                df.loc[date] = prediction

            # Plotting
            fig=plt.figure(figsize=(10, 6))
            plt.plot(data['date'], data['price'], label='Actual Data')
            plt.plot(future_dates, future_predictions, label='Future Predictions')

            plt.title('Time Series Prediction with LSTM')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(fig)
        else:
           st.write('RMSE VALUE OF GRU',gru_rmse)
            
           st.write('R2 score of GRU',gru_r2_score)


def Result():

    # Visualize the predictions and actual values
    fig=plt.figure(figsize=(20, 10))
    plt.title('Comparative Analysis of GRU and LSTM ')
    plt.plot(lstm_prediction['predictions'],color='green')
    plt.plot(lstm_prediction['Actuals'],color='black')
    plt.plot(gru_prediction['predictions'],color='orange')
    plt.legend(['LSTM_prediction', 'Actual_data','GRU_prediction'])
    st.pyplot(fig)

    st.write("RMSE values of LSTM and GRU")
    st.write('RMSE VALUE OF LSTM',lstm_rmse)
    st.write('RMSE score of GRU',gru_rmse)

    st.write('R2 scores of LSTM and GRU')
    st.write('R2 score of LSTM',lstm_r2_score)
    st.write('R2 score of GRU',gru_r2_score)




def stream(option):
    if(option=='Select'):
        st.write()
    elif(option=='Dataset'):
        st.write("Binance coin")
        op=st.selectbox('Select',('Past','Latest','Full dataset'))
        if(op=='Past'):
            st.write(data.head(10))
        elif(op=='Latest'):
            st.write(data.tail(10))
        elif(op=='Full dataset'):
            st.write(data)
        
    elif(option=='LSTM MODEL'):
        st.write('LONG SHORT-TERM MODEL')



        options = st.multiselect('select',['Train and Test', 'MSE and MAE', 'Actual and Predicted', 'Heat Map','RMSE value and R2 score','Future Prediction'])
        lstm_model_plot(options)
    elif(option=="GRU MODEL"):
        st.write('GRU')
        options = st.multiselect('select',['Train and Test', 'MSE and MAE', 'Actual and Predicted', 'Heat Map','RMSE value and R2 score','Future Prediction'])
        gru_model_plot(options)
    else:
        st.write('RESULT')
        Result()
stream(option)