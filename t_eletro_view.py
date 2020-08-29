from utils.elpv_reader import load_dataset
print('load_dataset...')
images, proba, types = load_dataset()

from keras.applications import MobileNet, VGG16
from keras.preprocessing.image import ImageDataGenerator
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(images.shape[1], images.shape[2], 3))

# Freeze only 4 last the layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Train the model. Here we will be using the imageDataGenerator for data augmentation.

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Change the batchsize according to your system RAM
train_batchsize = 5
val_batchsize = 5

# Data Generator for Training data
datagen.fit(images) 


# Compile the model
model.compile(loss='mean_squared_error',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['mse'])

# Train the Model
history = model.fit_generator(datagen.flow(images, proba, batch_size=1, subset='training'), 
			steps_per_epoch=len(images)*0.8/train_batchsize,
			validation_data=datagen.flow(images, proba, batch_size=1, subset='validation'),
            validation_steps=len(images)*0.2/val_batchsize,
            epochs=5, verbose=1)

# Save the Model
model.save('last4_layers.h5')

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
