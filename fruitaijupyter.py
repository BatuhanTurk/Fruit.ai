#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False)


# In[2]:


base_model.trainable = False


# In[3]:


inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation = 'softmax')(x)
model = keras.Model(inputs,outputs)


# In[4]:


model.summary()


# In[5]:


model.compile(loss = "categorical_crossentropy" , metrics = ["categorical_accuracy"])


# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        samplewise_center=True,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=True
)


# In[7]:


train_it = datagen.flow_from_directory("./fruits/train", 
                                       target_size=(224,224), 
                                       color_mode='rgb', 
                                       class_mode="categorical")

valid_it = datagen.flow_from_directory("./fruits/valid", 
                                      target_size=(224,224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")


# In[ ]:


model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=12)


# In[ ]:


base_model.trainable = True

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
              loss = "categorical_crossentropy" , metrics = ["categorical_accuracy"])


# In[ ]:


model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=12)


# In[ ]:


model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)


# In[ ]:


model.save("fruit_model.h5")

