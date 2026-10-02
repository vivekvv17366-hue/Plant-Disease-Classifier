import os 
import tensorflow as tf
import json
import matplotlib.pyplot as plt


from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



DATASET_PATH='datasets'
IMAGE_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=10
SEED=42


train_ds=tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds=tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names=train_ds.class_names

#print('class_names: ',class_names)
num_class=len(class_names)

# streamlit

with open('class_names.json','w') as f:
    json.dump(class_names,f)


data_agumentation=models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
    
AUTOTUNE= tf.data.AUTOTUNE

train_ds=train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.prefetch(buffer_size=AUTOTUNE)


base_model=MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable=False


inputs=layers.Input(shape=(224,224,3))

x=data_agumentation(inputs)

x = layers.Rescaling(
    scale=1./127.5,
    offset=-1
)(x)

x=base_model(x,training=False)

x=layers.GlobalAveragePooling2D()(x)

x=layers.Dropout(0.3)(x)

outputs=layers.Dense(num_class,activation='softmax')(x)

model=models.Model(inputs,outputs)


model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001),loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



train_result=model.fit(
    train_ds,validation_data=val_ds,epochs=EPOCHS
)


model.save('plant_disease_analyser.keras')
print('model saved successfully')


plt.figure(figsize=(8,5))

plt.plot(train_result.history['accuracy'],
         label='Training accuracy')
plt.plot(train_result.history['val_accuracy'],
         label='validation accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("Accuracy.png")
plt.show()


plt.figure(figsize=(8,5))

plt.plot(train_result.history['loss'],
         label='training loss')

plt.plot(train_result.history['validation loss'],
         label='validation loss')

plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.savefig('loss.png')
plt.show()


loss,accuracy=model.evaluate(val_ds)

print('validation loss:',loss)
print('validation accuracy: ',accuracy)