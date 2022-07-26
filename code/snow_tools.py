############# DEPENDENCIES ##############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

import os

from tensorflow import io
from tensorflow import strings
from tensorflow import argmax
from tensorflow.data import AUTOTUNE
from tensorflow.data import Dataset
from tensorflow.image import resize_with_pad

############# SETTING UP TENSORFLOW DATASETS FROM IMAGES ##############

def get_image(file_path):
    
    """
    This function reads in an image given a file path.  It uses the folder name containing each image to assign the image label.
    It assumes that the folder names are 'clear' and 'snow,' and it assigns the label 0 to 'clear' and 1 to 'snow.'
    It resizes images to 640x640 with a pad, meaning that no image in this dataset will be reduced in resolution.
    It scales the pixel values to the range from -1 to 1.
    """
    
    # Given a Tensorflow filepath object, sets label to 0 if the second to last part of the path is 'clear' and 1 if it is 'snow'.
    parts = strings.split(file_path, os.path.sep)
    label = parts[-2] == ['clear', 'snow']
    
    # Reads in the image at the given filepath, which should be a jpeg.
    img = io.read_file(file_path)
    img = io.decode_jpeg(img)
    
    # Add buffer as necessary to resize image to 640x640 without changing aspect ratio.
    # 640 is the largest dimesion of any image in the dataset.
    img = resize_with_pad(img, 640, 640)
    
    # Normalize images so each pixel is in the range -1 to 1
    return (img - 127.5) / 127.5, argmax(label)

def get_train_data(img_folder_list, shuffle_seed, train_size=0.8):
    
    """
    This function reads in the images from the train folder and generates Tensorflow Datasets.
    It assumes that the train data is split into two subfolders indicating their labels for the neural nets.
    """
    
    # Gets paths to all the files in the given folders holding the training data.  No shuffling is done yet.
    file_groups = [Dataset.list_files(folder+'*', shuffle=False) for folder in img_folder_list]
    
    train_num_0 = int(len(file_groups[0])*train_size) # Number of files in folder 0 to be used in training set
    train_num_1 = int(len(file_groups[1])*train_size) # Number of files in folder 1 to be used in training set
    
    # Select the files paths for training and validation from each folder.
    train_paths = file_groups[0].take(train_num_0).concatenate(file_groups[1].take(train_num_1))
    val_paths = file_groups[0].skip(train_num_0).concatenate(file_groups[1].skip(train_num_1))
    
    # Filepaths are now shuffled.  This will be redone in each epoch to randomize the order of images during training.
    train_paths = train_paths.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True, seed=shuffle_seed)
    val_paths = val_paths.shuffle(buffer_size=len(val_paths), reshuffle_each_iteration=True, seed=shuffle_seed)
    
    # Use the get_image function to import the images, which are resized and normalized as they are imported.
    train_images = train_paths.map(get_image, num_parallel_calls=AUTOTUNE)
    val_images = val_paths.map(get_image, num_parallel_calls=AUTOTUNE)
    
    return train_images, val_images

def get_test_data(img_folder_list, shuffle_seed):
    
    """
    This function reads in the images from the test folder and generates Tensorflow Datasets.
    It assumes that the test data is split into three subfolders with different snowflake sizes.
    It also assumes that each subfolder is further divided into two subfolders indicating their labels for the neural nets.
    """
    
    # Gets paths to all the files in the given folders holding the testing data.  No shuffling is done yet.
    file_groups = [Dataset.list_files(folder+'*', shuffle=False) for folder in img_folder_list]
    
    # Combine the 'clear' and 'snow' image paths of each size group.
    small_paths = file_groups[0].concatenate(file_groups[1])
    medium_paths = file_groups[2].concatenate(file_groups[3])
    large_paths = file_groups[4].concatenate(file_groups[5])
    
    # Filepaths within each size group are now shuffled.
    # There is no need to reshuffle after each iteration, since these images are only used once for each model.
    small_paths = small_paths.shuffle(buffer_size=len(small_paths), reshuffle_each_iteration=False, seed=shuffle_seed)
    medium_paths = medium_paths.shuffle(buffer_size=len(medium_paths), reshuffle_each_iteration=False, seed=shuffle_seed)
    large_paths = large_paths.shuffle(buffer_size=len(large_paths), reshuffle_each_iteration=False, seed=shuffle_seed)
    
    # Use the get_image function to import the images, which are resized and normalized as they are imported.
    small_images = small_paths.map(get_image, num_parallel_calls=AUTOTUNE)
    medium_images = medium_paths.map(get_image, num_parallel_calls=AUTOTUNE)
    large_images = large_paths.map(get_image, num_parallel_calls=AUTOTUNE)
    
    return small_images, medium_images, large_images

############# GENERATING AND SAVING MODEL PREDICTIONS AND TRUE VALUES ##############

def get_train_preds(data, model, model_name, which_data):
    
    """
    This fuction generates predictions on the given data using the given model.
    It outputs these predictions, along with true values, to a CSV.
    
    I use .predict_on_batch() here rather than .predict(), even though it's slower.
    This was the best way I found to match true and predicted values for the train and val data.
    The train and val datasets are set up to shuffle each iteration.
    This appears to prevent my labels and predictions from matching.
    """
    
    labels = []
    preds = []
    # Loop over each batch in the given data.
    for image_batch, label_batch in data:
        labels.append(label_batch.numpy()) # Append the true labels for the batch to a list of labels.
        preds.append(model.predict_on_batch(image_batch)[:,0]) # Append the model's predictions to a list.
    # Concatenate the lists of labels and predictions to numpy arrays, then store them in a DataFrame.
    results = pd.DataFrame(zip(np.concatenate(labels), np.concatenate(preds)), columns=['true', 'pred'])
    results.to_csv(f'../saved_models/{model_name}/{which_data}_preds.csv', index=False) # Save the Dataframe to a CSV.

def get_test_preds(data, model, model_name, which_data):
    
    """
    This fuction generates predictions on the given data using the given model.
    It outputs these predictions, along with true values, to a CSV.
    
    Since the test data has shuffle_each_iteration = False, I can use .predict(), which is faster than .predict_on_batch()
    """
    
    preds = model.predict(data)[:,0] # Generate a numpy array of predictions for the full dataset.
    labels = np.concatenate([label for image, label in data]) # Extract all true labels from the dataset.
    results = pd.DataFrame(zip(labels, preds), columns=['true', 'pred']) # Store the predictions and labels in a DataFrame.
    results.to_csv(f'../saved_models/{model_name}/{which_data}_preds.csv', index=False) # Save the DataFrame to a CSV.
    
############# CHECKING CLASSIFICATION METRICS ##############
    
def binarize_class(in_num, cutoff=0.5):
    
    """
    If input number < cutoff, return 0
    Otherwise, return 1
    Used by check_metrics()
    """
    
    if in_num < cutoff:
        return 0
    else:
        return 1

def check_metrics(df_dict, cat_0, cat_1, cutoff=0.5):
    
    """
    This function accepts a dictionary of DataFrames containing true and predicted values.
    It then displays several key classification metrics labeled according to the two categories provided.
    The function assumes that the predictions are provided as floats.
    It converts the preditions to 0 or 1 using the binarize_class() function.
    """
    
    # Convert predictions to 0 or 1 according to the given cutoff value.
    for key in df_dict:
        df_dict[key]['pred_binary'] = df_dict[key]['pred'].apply(binarize_class, args=(cutoff,))
    
    # For each DataFrame, print the classification report and ROC AUC score.
    for key in df_dict:
        print(f'{key}:\n')
        print(classification_report(df_dict[key]["true"], df_dict[key]["pred_binary"], target_names=[cat_0, cat_1], digits=4))
        print(f'ROC AUC score: {roc_auc_score(df_dict[key]["true"], df_dict[key]["pred_binary"])}')
        print('\n*************************\n')
    
    # Prepare a graph to display ROC curves and confusion matrices.
    fig, axs = plt.subplots(len(df_dict), 2, figsize = (15, len(df_dict)*5), gridspec_kw={'hspace': .3})
    
    # For each DataFrame, show the ROC curve and confusion matrix.
    for i, key in enumerate(df_dict):
        RocCurveDisplay.from_predictions(df_dict[key]['true'], df_dict[key]['pred_binary'], ax=axs[i,0])
        axs[i,0].set_title(f'{key} ROC curve', fontsize='x-large')
    
        ConfusionMatrixDisplay.from_predictions(df_dict[key]['true'], df_dict[key]['pred_binary'], display_labels=[cat_0, cat_1], ax=axs[i,1])
        axs[i,1].set_title(f'{key} confusion matrix', fontsize='x-large')