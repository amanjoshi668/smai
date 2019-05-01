
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import mean_squared_error


# In[5]:


from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[50]:


time_series = pd.read_csv("GoogleStocks.csv", parse_dates=True)
time_series = time_series.drop(time_series.index[0])


# In[51]:


time_series.head()


# In[52]:


time_series['high'] = pd.to_numeric(time_series['high'])
time_series['low'] = pd.to_numeric(time_series['low'])
time_series['average'] = (time_series['high'] + time_series['low']) / 2
time_series = time_series.sort_values('date')
time_series.head()


# In[53]:


columns = list(time_series.columns.values)
del columns[0]
fig = plt.figure()
plt.style.use('fivethirtyeight')
for i, column in enumerate(columns):
    plt.subplot(len(columns), 1, i + 1)
    plt.plot(time_series.index, time_series[column])
    plt.title(column, y=0.5, loc='right')
plt.show()


# In[11]:


time_series.isna().any()
data = time_series[['open', 'average', 'volume']]
# Feature Scaling Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(data)


# In[ ]:


def create_dataset(data, time_stamp):
    X_train = []
    y_train = []
    print("Time stamp: {0}".format(time_stamp))
    for i in range(time_stamp, len(data)):
        X_train.append(
            np.array([data[i - time_stamp:i, 1], data[i - time_stamp:i, 2]]).T)
        y_train.append(data[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train[:-100], y_train[:-100], X_train[-100:], y_train[-100:]


# In[ ]:


def create_model(number_of_layer, number_of_cells, input_shape):
    model = Sequential()
    model.add(
        LSTM(
            units=number_of_cells,
            return_sequences=True,
            input_shape=input_shape))
    input_shape = None
    model.add(Dropout(0.2))
    for l in range(number_of_layer - 1):
        return_sequence = True
        if (l == number_of_layer - 2):
            return_sequence = False
        model.add(
            LSTM(units=number_of_cells, return_sequences=return_sequence))
        model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[ ]:


def run_model(data, number_of_layer, number_of_cells, time_stamp):
    X_train, y_train, X_test, y_test = create_dataset(data, time_stamp)
    print("X train shape: {0}, y_train shape: {1}".format(
        X_train.shape, y_train.shape))
    model = create_model(number_of_layer, number_of_cells, (time_stamp, 2))
    print(model.summary())
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    predicted_open_price = model.predict(X_test)
    print(
        "Number of layer: {0}, Number of cells: {1}, time_stamp: {2}, Mean square error :{3}"
        .format(number_of_layer, number_of_cells, time_stamp,
                mean_squared_error(predicted_open_price, y_test)))
    plt.plot(
        range(1, 1 + len(predicted_open_price)),
        predicted_open_price,
        label="predicted price")
    plt.plot(
        range(1, 1 + len(predicted_open_price)),
        y_test,
        label="original price")
    plt.legend(loc="best")
    plt.show()
    return predicted_open_price


# In[ ]:


run_model(training_set_scaled, 2, 30, 20)


# In[ ]:


for num_layers in [2, 3]:
    for num_cells in [30, 50, 80]:
        for time_step in [20, 50, 75]:
            run_model(training_set_scaled, num_layers, num_cells, time_step)


# ## Question 1-2
# ## HMM

# In[ ]:


def run_hmm(number_of_hidden_states, time_step):
    predicted = []
    actual = []

    i = 753
    model = hmm.GaussianHMM(
        n_components=number_of_hidden_states,
        covariance_type='full',
        init_params='stmc')
    while (i > 653):
        test_data = training_set_scaled[i]
        train_data = training_set_scaled[:i, :][::-1]
        model.fit(train_data)

        curr_likelihood = model.score(train_data[0:time_step - 1, :])
        past_likelihood = []
        iters = 1
        while iters < len(train_data) / time_step - 1:
            past_likelihood = np.append(
                past_likelihood,
                model.score(train_data[iters:iters + time_step - 1, :]))
            iters = iters + 1
        absolute_liklehood = np.absolute(past_likelihood - curr_likelihood)
        likelihood_diff_i = np.argmin(absolute_liklehood)
        predicted_change = train_data[likelihood_diff_i, :] - train_data[
            likelihood_diff_i + 1, :]
        predicted.append(test_data + predicted_change)
        actual.append(test_data)
        i -= 1
    predicted = np.array(predicted)[:, 0]
    actual = np.array(actual)[:, 0]
    plt.plot(predicted, "-D", label="predicted")
    plt.plot(actual, "-D", label="actual")
    print(
        "Number of cells: {0}, time_step: {1}, Mean square error :{2}".format(
            number_of_hidden_states, time_step,
            mean_squared_error(predicted, actual)))
    plt.legend(loc="best")
    plt.show()
    return predicted


# In[ ]:


for number_of_hidden_states in [4, 8, 12]:
    for time_step in [20, 50, 75]:
        run_hmm(number_of_hidden_states, time_step)


# ## Question 1-3

# In[20]:


hmm_predicted_price = run_hmm(12, 75)[::-1]
rnn_predicted_price = run_model(training_set_scaled, 3, 80, 75)
actual_price = training_set_scaled[-100:, 0]
plt.plot(actual_price, "-D", label="Actual price")
plt.plot(hmm_predicted_price, "-D", label="HMM predicted price")
plt.plot(rnn_predicted_price, "-D", label="RNN predicted price")
plt.legend(loc="best")
plt.show()


# ## Question 2

# In[54]:


prob_emmision = {
    'E': {
        'A': 0.25,
        'C': 0.25,
        'G': 0.25,
        'T': 0.25
    },
    '5': {
        'A': 0.05,
        'C': 0,
        'G': 0.95,
        'T': 0
    },
    'I': {
        'A': 0.4,
        'C': 0.1,
        'G': 0.1,
        'T': 0.4
    }
}


# In[55]:


prob_transition = {
    '^': {
        'E': 1
    },
    'E': {
        'E': 0.9,
        '5': 0.1
    },
    '5': {
        'I': 1
    },
    'I': {
        'I': 0.9,
        '$': 0.1
    }
}


# In[56]:


sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"
state_path = "EEEEEEEEEEEEEEEEEE5IIIIIII$"


# In[57]:


prob = 1
for i in range(len(sequence)):
    s0 = state_path[i]
    s1 = state_path[i + 1]
    n = sequence[i]
    e = prob_emmision[s0][n]
    t = prob_transition[s0][s1]
    prob = prob * e * t
print(prob)


# In[58]:


np.log(prob)

