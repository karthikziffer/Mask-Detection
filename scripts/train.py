import warnings
warnings.filterwarnings("ignore")

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras


batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = True,
        class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
        './test',
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle = True,
        class_mode='binary')


base_model = keras.applications.MobileNetV2(
    input_shape= (150, 150, 3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    pooling='avg')

for layer in base_model.layers[:10]:
  layer.trainable = False

x = base_model.output
out_pred = Dense(1, activation= "sigmoid")(x)
model = Model(inputs = base_model.input, outputs= out_pred)
opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        epochs=30,
        validation_data=validation_generator)