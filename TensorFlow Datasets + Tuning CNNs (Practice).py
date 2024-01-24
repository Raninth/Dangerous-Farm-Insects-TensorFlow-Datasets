#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import tensorflow as tf

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for TensorFlow
tf.random.set_seed(42)

import os, glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
tf.__version__


# In[42]:


import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# In[44]:


import os

data_dir = "C:\\Users\\dell\\Desktop\\farm_insects"

# Get the list of files and subdirectories in the "farm_insects" folder
contents = os.listdir(data_dir)

# Print the contents
print(contents)


# In[45]:


import glob

# Getting list of img file paths (ONLY, make it recursive to include subdirectories)
img_files = glob.glob(data_dir + "/**/*", recursive=True)
len(img_files)


# In[46]:


# Take a look at the first 5 filepaths
img_files[0:5]



# In[47]:


# Gettting the list of folders from data dir
subfolders = os.listdir(data_dir)
subfolders


# In[49]:


# Saving image params as vars for reuse
batch_size = 32
img_height = 128
img_width = 128


# In[50]:


# make the dataset from the main folder of images
ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=True,
    label_mode="categorical",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

ds


# In[51]:


len(ds)


# In[52]:


#Save the class_names and number of classes as variables to reuse later.

class_names = ds.class_names
class_names



# In[53]:


# Saving # of classes for reuse
num_classes = len(class_names)
num_classes



# In[78]:


# Saving dictionary of integer:string labels
class_dict = dict(zip(range(num_classes), class_names))
class_dict



# In[54]:


##Split the dataset into a 70% training, 20 % validation, and 10% test split.
## Set the size of the 
split_train = 0.7
split_val = 0.2
split_test = 0.1

# Calculate the number of batches for training and validation data 
n_train_batches =  int(len(ds) * split_train)
n_val_batches = int(len(ds) * split_val)

print(f"Use {n_train_batches} batches as training data")
print(f"Use {n_val_batches} batches as validation data")
print(f"The remaining {len(ds)- (n_train_batches+n_val_batches)} batches will be used as test data.")


# In[55]:


# Use .take to slice out the number of batches 
train_ds = ds.take(n_train_batches)
# Confirm the length of the training set
len(train_ds)


# In[56]:


# Skipover the training batches
val_ds = ds.skip(n_train_batches)
# Take the correct number of validation batches
val_ds = val_ds.take(n_val_batches)
# Confirm the length of the validation set
len(val_ds)




# In[57]:


# Skip over all of the training + validation batches
test_ds = ds.skip(n_train_batches + n_val_batches)
# Confirm the length of the testing data
len(test_ds)



# In[58]:


# Determine number of batches in dataset
ds_size = len(ds)
ds_size


# ## Preview the Data

# In[59]:


# checking the class names
class_names = ds.class_names

class_dict = dict(zip(range(len(class_names)), class_names))
class_dict


# In[60]:


# Batch Size
batch_size


# In[62]:


# taking a sample banch to see batch shape
example_batch_imgs,example_batch_y= train_ds.take(1).get_single_element()
example_batch_imgs.shape


# In[63]:


# individual image shape
# individual image shape
input_shape = example_batch_imgs[0].shape
input_shape


# In[65]:


array_to_img(example_batch_imgs[0])


# In[66]:


[*input_shape]


# ## Optimize Dataset Performance

# In[67]:


# Use autotune to automatically determine best buffer sizes 
AUTOTUNE = tf.data.AUTOTUNE

# ORIGINAL SHUFFLED TRAIN
train_ds = train_ds.cache().shuffle(buffer_size= len(train_ds),
                                   seed=42).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ## Build a Simple CNN Model
# 
# 

# In[68]:


def build_model1(name="CNN1", input_shape=input_shape):
    model = models.Sequential(name=name)
    # Using rescaling layer to scale pixel values
    model.add(layers.Rescaling(1.0 / 255, input_shape=input_shape))

    # Convolutional layer
    model.add(
        layers.Conv2D(
            filters=16,  # How many filters you want to use
            kernel_size=3,  # size of each filter
            input_shape=input_shape,
            padding="same",
        )
    )
    # Pooling layer
    model.add(layers.MaxPooling2D(pool_size=2))  # Size of pooling

    # Convolutional layer
    model.add(
        layers.Conv2D(
            filters=32,  # How many filters you want to use
            kernel_size=3,  # size of each filter
            input_shape=input_shape,
            padding="same",
        )
    )
    # Pooling layer
    model.add(layers.MaxPooling2D(pool_size=2))  # Size of pooling

    # Flattening layer
    model.add(layers.Flatten())
    # Output layer
    model.add(
        layers.Dense(
            len(class_names), activation="softmax"
        )  # How many output possibilities we have
    )  # What activation function are you using?

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ## Fit the Model
# 

# In[69]:


# Build fresh model and train
model1 = build_model1(name="Model1")

# fit the neural network
history = model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)


# ## Evaluate the Model
# 

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_history(history,figsize=(6,8)):
    # Get a unique list of metrics 
    all_metrics = np.unique([k.replace('val_','') for k in history.history.keys()])

    # Plot each metric
    n_plots = len(all_metrics)
    fig, axes = plt.subplots(nrows=n_plots, figsize=figsize)
    axes = axes.flatten()

    # Loop through metric names add get an index for the axes
    for i, metric in enumerate(all_metrics):

        # Get the epochs and metric values
        epochs = history.epoch
        score = history.history[metric]

        # Plot the training results
        axes[i].plot(epochs, score, label=metric, marker='.')
        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}",marker='.')
        except:
            pass

        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch",ylabel=metric)

    # Adjust subplots and show
    fig.tight_layout()
    plt.show()


def evaluate_classification(model, X_train, y_train, X_test, y_test,
                         figsize=(6,4), normalize='true', output_dict = False,
                            cmap_train='Blues', cmap_test="Reds",colorbar=False):

    # Get predictions for training data
    y_train_pred = model.predict(X_train)

    # Call the helper function to obtain regression metrics for training data
    results_train = classification_metrics(y_train, y_train_pred, #verbose = verbose,
                                     output_dict=True, figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_train,
                                     label='Training Data')
    print()
    # Get predictions for test data
    y_test_pred = model.predict(X_test)
    # Call the helper function to obtain regression metrics for test data
    results_test = classification_metrics(y_test, y_test_pred, #verbose = verbose,
                                  output_dict=True,figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_test,
                                    label='Test Data' )
    if output_dict == True:
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train,
                    'test': results_test}
        return results_dict


### FINAL FROM FLEXIBILE EVAL FUNCTIONS LESSON

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def classification_metrics(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f"):
    """Modified version of classification metrics function from Intro to Machine Learning.
    Updates:
    - Reversed raw counts confusion matrix cmap  (so darker==more).
    - Added arg for normalized confusion matrix values_format
    """
    # Get the classification report
    report = classification_report(y_true, y_pred)

    ## Print header and report
    header = "-"*70
    print(header, f" Classification Metrics: {label}", header, sep='\n')
    print(report)

    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")


    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")

    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()

    # Return dictionary of classification_report
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict


##########

def get_true_pred_labels(model,ds):
    """Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
    Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    """
    y_true = []
    y_pred_probs = []

    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():

        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)

        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)

    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    return y_true, y_pred_probs


def convert_y_to_sklearn_classes(y, verbose=False):
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y

    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y has is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)

    else:
        if verbose:
            print("y has 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)


def evaluate_classification_network(model, 
                                    X_train=None, y_train=None, 
                                    X_test=None, y_test=None,
                                    history=None, history_figsize=(6,6),
                                    figsize=(6,4), normalize='true',
                                    output_dict = False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                    colorbar=False):
    """Evaluates a neural network classification task using either
    separate X and y arrays or a tensorflow Dataset

    Data Args:
        X_train (array, or Dataset)
        y_train (array, or None if using a Dataset
        X_test (array, or Dataset)
        y_test (array, or None if using a Dataset)
        history (history object)
        """
    # Plot history, if provided
    if history is not None:
        plot_history(history, figsize=history_figsize)

    ## Adding a Print Header
    print("\n"+'='*80)
    print('- Evaluating Network...')
    print('='*80)


    ## TRAINING DATA EVALUATION
    # check if X_train was provided
    if X_train is not None:
        ## Check if X_train is a dataset
        if hasattr(X_train,'map'):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_train, y_train_pred = get_true_pred_labels(model, X_train)
        else:
            # Get predictions for training data
            y_train_pred = model.predict(X_train)

        ## Pass both y-vars through helper compatibility function
        y_train = convert_y_to_sklearn_classes(y_train)
        y_train_pred = convert_y_to_sklearn_classes(y_train_pred)

        # Call the helper function to obtain regression metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, 
                                         output_dict=True, figsize=figsize,
                                             colorbar=colorbar, cmap=cmap_train,
                                               values_format=values_format,
                                         label='Training Data')

        ## Run model.evaluate         
        print("\n- Evaluating Training Data:")
        print(model.evaluate(X_train, return_dict=True))

    # If no X_train, then save empty list for results_train
    else:
        results_train = []


    ## TEST DATA EVALUATION
    # check if X_test was provided
    if X_test is not None:
        ## Check if X_train is a dataset
        if hasattr(X_test,'map'):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_test, y_test_pred = get_true_pred_labels(model, X_test)
        else:
            # Get predictions for training data
            y_test_pred = model.predict(X_test)

        ## Pass both y-vars through helper compatibility function
        y_test = convert_y_to_sklearn_classes(y_test)
        y_test_pred = convert_y_to_sklearn_classes(y_test_pred)

        # Call the helper function to obtain regression metrics for training data
        results_test = classification_metrics(y_test, y_test_pred, 
                                         output_dict=True, figsize=figsize,
                                             colorbar=colorbar, cmap=cmap_test,
                                              values_format=values_format,
                                         label='Test Data')

        ## Run model.evaluate         
        print("\n- Evaluating Test Data:")
        print(model.evaluate(X_test, return_dict=True))

    # If no X_test, then save empty list for results_test
    else:
        results_test = []


    # Store results in a dataframe if ouput_frame is True
    results_dict = {'train':results_train,
                    'test': results_test}
    return results_dict


# In[71]:


evaluate_classification_network(model1, X_test=test_ds, history=history);


# # Tuning CNNs

# In[72]:


def build_model2(hp):
    
    model = models.Sequential(name='TunedModel')
    # Using rescaling layer to scale pixel values
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    
    # Convolutional layer
    model.add(
        layers.Conv2D(
            filters=hp.Int(name='filters_1', 
                           min_value=32, 
                           max_value=64,
                           step=16),  # How many filters you want to use
            kernel_size=3,  # size of each filter
            input_shape=input_shape,
            padding='same'))
    
    # Pooling layer
    pool_padding = 'same'
    pool_strides = 2
    model.add(layers.MaxPooling2D(pool_size=2, 
                                  padding=pool_padding,
                                 strides=pool_strides))  # Size of pooling


    # Convolutional layer
    model.add(
        layers.Conv2D(
            filters=hp.Int(name='filters_2', 
                           min_value=32, 
                           max_value=64,
                           step=32),  # How many filters you want to use
            kernel_size=3,  # size of each filter
            input_shape=input_shape,
            padding='same')) 

    # Pooling layer
    model.add(layers.MaxPooling2D(pool_size=2, 
                                  padding=pool_padding,
                                 strides=pool_strides))  # Size of pooling




    # if hp.Boolean('batch_normalization'):
    #     model.add(layers.BatchNormalization())
        
    
    # Flattening layer
    model.add(layers.Flatten())

    # Tune n_dense layers
    n_layers = hp.Int('n_hidden_dense_layers', 
                          min_value=1,
                          max_value=3)
    n_units = hp.Float('n_units_dense',min_value=128,max_value=1024,step=128,)
    dropout_rate = hp.Float("drop_rate",min_value=0.0,
                                         max_value=0.5, step=0.1)
    for n in range(n_layers):
        model.add(layers.Dense(n_units, activation='relu'))       
        # Add dropout
        model.add(layers.Dropout(dropout_rate))
    
        
        
    # Output layer
    model.add(
        layers.Dense(len(class_names), activation="softmax")  # How many output possibilities we have
    )  # What activation function are you using?

    # ## Tuning for Optimizer (classes)
    # chosen_optimizer = hp.Choice('optimizer',['adam','nadam','rmsprop'])
    # if chosen_optimizer=='adam':
    #     optimizer = tf.keras.optimizers.legacy.Adam()
    # elif chosen_optimizer=='nadam':
    #     optimizer =tf.keras.optimizers.legacy.Nadam()
    # elif chosen_optimizer=='rmsprop':
    #     optimizer = tf.keras.optimizers.legacy.RMSprop()

    

    ## Compile model   
    lr = hp.Float('learning_rate',min_value=.0001, max_value=10, step=10, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy',])
    
    model.summary()
    return model


# In[73]:


def get_callbacks( ):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5,
                                                      monitor='val_accuracy')
    
    return [early_stopping]


# In[74]:


get_ipython().system('pip install visualkeras')


# In[75]:


get_ipython().system('pip install keras_tuner')


# In[76]:


get_ipython().system('pip install tensorflow')


# In[77]:


import numpy as np
import tensorflow as tf
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for TensorFlow
tf.random.set_seed(42)

import os, glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import visualkeras as vk

import keras_tuner as kt
from keras_tuner import HyperParameters as hp

import os

folder = 'KerasTuner/'
os.makedirs(folder, exist_ok=True)


# In[22]:


get_ipython().system('pip')


# In[78]:


get_ipython().system('pip install keras-tuner')


# In[79]:


import keras_tuner as kt
from keras_tuner import HyperParameters as hp

import os
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
import os



# In[80]:


#hyperband allows for callbacks
tuner_hb = kt.Hyperband(build_model2, objective='val_accuracy',
                        max_epochs=25, overwrite=True, directory=folder, 
                project_name='tuning-cnn1',)
tuner_hb.search_space_summary()


# In[81]:


tuner_hb.search(train_ds,validation_data=val_ds, epochs=20)#, callbacks=get_callbacks())
tuner_hb.results_summary()


# In[82]:


best_hps = tuner_hb.get_best_hyperparameters()[0]
print(best_hps.values)


# In[83]:


best_model = tuner_hb.get_best_models()[0]
best_model.summary()


# In[84]:


evaluate_classification_network(best_model, X_test=test_ds, history=history);


# In[ ]:




