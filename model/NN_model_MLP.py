import math
import numpy as np

from keras import optimizers
from keras.layers import Dense, Flatten, Embedding, SpatialDropout1D, Dropout, regularizers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from utils.metrics import Metrics
from services.data_preprocessor.clean_data import *
from keras.callbacks import LearningRateScheduler


def step_decay(epoch):
    initial_l_rate = learning_rate
    drop = 0.5
    epochs_drop = 10.0
    l_rate = initial_l_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return l_rate


data = clean_and_store_data(training_data_set)

input_data = data
input_data = data_to_sequence(input_data)
X = pad_sequences(input_data, maxlen=4, padding='post')
Y = pd.get_dummies(data[Facing_key])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Training data shape: {0} {1}- validation data shape: {2} {3}'.format(
    X_train.shape,
    Y_train.shape,
    X_test.shape,
    Y_test.shape))


l_rate = LearningRateScheduler(step_decay)
metrics = Metrics.Metrics(training_data=(X_train, Y_train))

callbacks_list = [l_rate, metrics]

global model
model = Sequential()
optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=False)
model.add(Embedding(300, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
plot_model(model,
           show_shapes=True,
           show_layer_names=True,
           to_file='model.png')

# **********************************************************************************************
History = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test, Y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks_list,
    verbose=2)

to_csv_file = list()
print('Id, Facing')
data = clean_and_store_data(testing_data_set)
for index, row in data.iterrows():
    X = pad_sequences([data_from_row(row)], maxlen=4, padding='post')
    predicted_output = model.predict(X, batch_size=1, verbose=2)
    predicted_output = convert_numeric_to_facing( str(np.argmax(predicted_output[0]).item()) )
    #print(str(row[Id_key])+', '+str(predicted_output))
    to_csv_file.append((row[Id_key], predicted_output))

#print(to_csv_file)
#df = pd.DataFrame(to_csv_file, columns=['Id', 'Facing'])
#print(df)

#df.to_csv(final_output_dataset, index=False)
