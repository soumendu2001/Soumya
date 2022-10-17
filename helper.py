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

  # Create and compile a Tensorflow Hub Feature Extractor
  def create_compile_model(model_url, num_classes, IMG_SIZE =224):
    import tensorflow_hub as hub
    import tensorflow as tf
    from tensorflow.keras import layers
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
    model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics =['accuracy'))
    return model                                                   
                                                                                                       
