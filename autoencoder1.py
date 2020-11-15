import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras import regularizers

train = pd.read_csv("./fashion-mnist_train.csv")
train_x = train[list(train.columns)[1:]].values
train_y = train['label'].values

## normalize and reshape the predictors
train_x = train_x / 255

## create train and validation datasets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

## reshape the inputs
train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)

input_layer = Input(shape=(784,))

encoding_layer_1 = Dense(300, activation="relu")(input_layer)
encoding_layer_2 = Dense(100, activation="relu")(encoding_layer_1)

latent_layer = Dense(10, activation='softmax')(encoding_layer_2)

decoding_layer1 = Dense(100, activation='relu')(latent_layer)
decoding_layer2 = Dense(300, activation='relu')(decoding_layer1)


output_layer = Dense(784)(decoding_layer2)

model = Model(input_layer, output_layer)
model.summary()
plot_model(model, to_file='autoencoder_vanilla.png', show_shapes=True)
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
model.fit(train_x, train_x, epochs=30, batch_size=200, validation_split=0.2, callbacks=[early_stopping])

np.random.shuffle(val_x)
predictions = model.predict(val_x)


def show_res(pred_list, act_list):
    n = len(pred_list)

    _, axes = plt.subplots(n, 2)
    for i, (ax, img_a, img_p) in enumerate(zip(axes, act_list, pred_list)):
        pixels_actual = act_list[i]
        pixels_predicted = pred_list[i]
        fig_resolution = (28, 28)
        pixels_a = pixels_actual.reshape(fig_resolution)
        pixels_p = pixels_predicted.reshape(fig_resolution)
        ax[0].imshow(pixels_a)
        ax[1].imshow(pixels_p)


show_res(predictions[0:6, :], val_x[0:6, :])
plt.show()
