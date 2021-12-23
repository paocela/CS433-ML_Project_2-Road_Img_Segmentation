from data_loading import *
from save_output import *
from model import Unet
from loss_functions import *
import tensorflow as tf

# initial data
testset_data_filename = 'Data/test_set_images/'
TEST_SIZE = 50 # number of images to be used
new_size = 608
image_shape = (400, 400, 3)
n_classes = 1
n_filters = 32
weights_prefix = "/Weights/model-unet-"

"""Load and predict testset data"""
def main(argv):

  # TODO: complete it
  # use requested loss function
  if argv[1] == "-msl":
    loss = tf.keras.losses.MeanSquaredLogarithmicError
    weights_path = weights_prefix + "MSL" + ".h5"
  elif argv[1] == "-focal":
    loss = FocalLoss
    weights_path = weights_prefix + "focal" + ".h5"
  elif argv[1] == "-bce":
    loss = 
    weights_path = weights_prefix + "BCE" + ".h5"
  else:
    raise Exception("please use as argument only the following: -msl, -focal, -bce")

  # Extract test data into numpy arrays.
  print("loading test data...")
  testset_data = extract_test_data(testset_data_filename, TEST_SIZE)

  """define model to be used"""
  input = Input(image_shape)

  # create UNET object
  unet_object = Unet(n_classes, n_filters)

  # apply keras.Input object to model and obtain outputs
  output = unet_object.forward(input)

  # groups layers into a model object with training and inference features
  model = Model(inputs=[input], outputs=[output])

  # compile model for training
  print("compiling UNET model...")
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=loss,
                metrics=[keras.metrics.Precision(), keras.metrics.Recall(), "accuracy"])

  print("model summary:")
  print(model.summary())

  """calculate and save predictions"""
  # load precomputed weights for model
  model.load_weights(weights_path)

  # predict using previously defined model
  print("calculating predictions using UNET model...")
  pred_testset = model_predict(model, testset_data, new_size)

  # save predictions to output file
  print("saving predictions...")
  masks_to_submission_outer('Output/submission_MSL.csv', pred_testset)

  print("Done!")

if __name__ == '__main__':
    main(sys.argv)