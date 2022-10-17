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
  plt.plot(epochs, training_accuracy, label ="Training_Accuracy")
  plt.plot(epochs, validation_accuracy, label ="Validation Accuracy") 
  plt.xlabel("epochs")
  plt.ylabel("Accuracy")
  plt.title("Accurac")
  plt.legend()
