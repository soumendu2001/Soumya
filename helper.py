# Plot validation data and training history seperately in a plot.
import matplotlib.pyplot as plt
def plot_loss_curve(history):
  """
  Returns seperate loss and accuracy curves for training and valiation metrics.
  Args:
  history: Tensorflow model history object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  
  """
  training_loss = history.history["loss"]
  validation_loss = history.history["val_loss"]
  training_accuracy = history.history["accuracy"]
  validation_accuracy = history.history["val_accuracy"]
  
  epochs = range(len(history.history["loss"]))
  # Plot Loss
  plt.plot(epochs,training_loss, label ="Training_Loss")
  plt.plot(epochs, validation_loss, label ="Validation_Loss")
  plt.xlabel("epochs")
  plt.ylabel("Loss")
  plt.title("Loss")
  plt.legend()
  
  #Plot accuracy
  plt.figure()
  plt.plot(epochs, training_accuracy, label ="Training_Accuracy")
  plt.plot(epochs, validation_accuracy, label ="Validation Accuracy") 
  plt.xlabel("epochs")
  plt.ylabel("Accuracy")
  plt.title("Accuracy")
  plt.legend()
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers
# Create and compile a Tensorflow Hub Feature Extractor
def create_model(model_url, num_classes, IMG_SIZE =224):
 
  """
  takes a tensorflow hub feature extractor mode URl, creates,compiles and returns it back.
  Input:
  model_url ( String) : URL of the tensorflow HUB feature extractor model.
  num_classes ( int) : Number of output neurons in the output_layer, should be equal to the number of classes.
  IMG_SIZE ( int):  Input size of the feature extractor model .default size is 224 having 3 color channel.
  Output:
  A compiled Keras Sequantial layer with model_url as feature extractor and Dense output layer as number of classes output neuron.
  """
  # Download the pre-trained model and save it as a Keras Layer
  feature_extractor_layer = hub.KerasLayer(model_url, trainable = False, name ='feature_extractor_layer', input_layer = (IMG_SIZE,IMG_SIZE,3))
  #Create our own model.
  model = tf.keras.Sequential([ feature_extractor_layer, layers.Dense(10,activation = "softmax", name = "Output_Layer")])
  #Compile the model.
  model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics =['accuracy'])
  return model
                                                                                                       
#Create a function to unzip a zip file into current working directory
# Since we are downloading and unzipping few files.
import zipfile
def unzip_data(filename):
  """
  Unzips filename in the current working directory
  Args:
  Input:
  filename ( str ): it is the filename path to the zip file which we need to unzip.
  """
  
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()
import os
# Create a function to walk-through an image classification directory and calculate how many image files are available.
def walk_through(dir_path):
  """
  walks through dir_path and calculates the number of image files in its folders and sub-folders
  Input:
  dir_path (str) : Target directory name
  Output:
  returns a printout of
  Number of subdirectories in each path
  Number of images ( files ) in each directory.
  name of each subdirectory.
  """
  for dirpath, dirname , filenames in os.walk(dir_path):
    print(f"There are {len(dirname)} directories and {len(filenames)} files in {dirpath}")
def load_and_prep(file_name, img_size = 224,scale = True):
  """
  loads and prepares and image mentioned in the filename to the size (224,224,3)
  input:
  file_name(str): Filename / path of the image
  img_size (int): Dimension of the image to be specified and returned. By default it is 224.
  scale (Boolean): To confirm if the image is scaled. default is True.
  Output:
  prepared image with the dimension (224,224,3)
  """
  #Read the file
  img = tf.io.read_file(file_name)
  #Decode the file
  img = tf.image.decode_jpeg(img)
  #resize the image
  img = tf.image.resize(img,[img_size, img_size])
  if scale:
    return img/255.
  else:
    return img
# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
# Function to evaluate: accuracy, precision, recall, f1-score
def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  from sklearn.metrics import accuracy_score, precision_recall_fscore_support
  print("Showing modified function")
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results   
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.
  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
