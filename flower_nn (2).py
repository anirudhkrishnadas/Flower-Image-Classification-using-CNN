import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

#reading the flower input and target files. 
print("Reading flowers input file.")
images = np.load("50x50flowers_images.npy")
print("Reading flowers target file.")
targets = np.load("50x50flowers_targets.npy")

#Spliting the input and target into 80% training and 20 % testing. 
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

#A deep convolutational neural network and three convoluation and maxpooing layer. 
#Dropout used to avoid overfittting. 
#i tried smaller network by reducing the number of convolution and dense layer and it felt like the model was not learning for it. 
def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=(50, 50, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.65))
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.65))
    model.add(layers.Dense(17, activation="softmax"))
    return model

model = get_model()
print("Training network.")

#I have used adam as optimizer and used the loss function as sparse_categorical cross entropy.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#early stopping used to monitor validaiton loss to control training. 
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, restore_best_weights=True)

#the batch size and other hyper parmeters were tuned by trial and error. Aiming to reduce the overfiting
# and making sure the learning happens properly.  

history = model.fit(X_train, y_train, epochs=1000,
                    batch_size=1024,
                    validation_split=0.2,
                    callbacks=[early_stopping]
                    ,verbose = 0)

#the training score and the testing score are displayed to the user. 
training_score = [history.history['loss'][-1], history.history['accuracy'][-1]]
print("The training score is", training_score)

testing_score = model.evaluate(X_test, y_test,verbose=0)
print("The testing score is", testing_score)


#saves the stopped epoch value and then ploting it with training loss.
stop = early_stopping.stopped_epoch
epoch= np.arange(stop + 1)
plt.plot(epoch, history.history['loss'])
plt.grid(axis='y', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.grid(axis='x', linestyle='-')

#this was the result, i got. 
#The training score is [1.1867313385009766, 0.7195402383804321]
#The testing score is [1.7000349760055542, 0.6213235259056091]

#I have some experience in working with ANN, so i used to try to bring the overfiting down. 