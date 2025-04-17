import argparse
import os
from datetime import datetime as dt

import tensorflow as tf
from masters.models.armnet import RMFNet
from tensorflow.keras.metrics import F1Score
# from masters.models.armnet import ARMNet, RNe, MyModel, RMFNet



def preprocess_image(image, label):
    # Check if the image has 3 channels (RGB)
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    return image, label


def main():
  #Very  useful: https://www.tensorflow.org/tutorials/load_data/images?hl=pt-br
  # Can apply custom logic to resize: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing
  #   Sticking to default for simplicity
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input',  type=str, default='data/mbtd/', help='input')
  parser.add_argument('-o', '--output', default='.', type=str, help='output')
  parser.add_argument('-s', '--seed', default=123, type=int, help='output')
  parser.add_argument('-e', '--epochs', default=50, type=int, help='output')
  parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
  parser.add_argument('-d', '--development', help='changes to dev experience', action='store_true')
  parser.add_argument('--multichoice', choices=['a', 'b', 'c'], nargs='+', type=str, help='multiple types of arguments. May be called all at the same time.')
  args = parser.parse_args()

  # Method parameters
  input_shape = (224, 224)
  batch_size = 64
  train_folder = os.path.join(args.input, "raw/Training")
  test_folder  = os.path.join(args.input, "raw/Testing")
  epochs = args.epochs
  validation_split_size = 0.2
  seed = args.seed

  # Load model
  model = RMFNet()
  # model.save would save the model with weights

  # Parameters
  img_height, img_width = input_shape

  # Load data
  train_ds = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    validation_split=validation_split_size,
    subset="training",
    seed=seed,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)#.take(10)
  
  val_ds = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    validation_split=validation_split_size,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size)#.take(10)

  test_ds = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    seed=seed,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size)#.take(10)
  
  # Reduces input size when in development mode
  if args.development:
    train_ds = train_ds.take(10)
    val_ds = val_ds.take(10)
    test_ds = test_ds.take(10)
    epochs = 1
    stage = "dev"
  else: 
    stage = "prod"

  # Apply preprocessing to standardize images
  # train_ds = train_ds.map(preprocess_image)
  # val_ds = val_ds.map(preprocess_image)

  # print(train_ds)
  # print(val_ds)


  # Defines the checkpoints
  #   If not exists, creates the directory
  checkpoint_path = f"experiments/model_checkpoints/armnet/{stage}/"+"armnet-{epoch:04d}.weights.h5"
  # checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
  # Save the last one and the best one
  
  f1 = F1Score(average='macro', threshold=0.5)
  precision_metric = tf.keras.metrics.Precision(name = 'precision')#, class_id = 4)

  model.compile(
    optimizer='adam',
    # loss='categorical',
    loss='categorical_crossentropy',
    # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), #'categorical_crossentropy', #
    metrics=['accuracy', precision_metric, f1])# F1Score()
  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs, 
    callbacks=[cp_callback]
  )

  print(history)
  print("Evaluate")
  result = model.evaluate(test_ds)
  print(dict(zip(model.metrics_names, result)))

if __name__ == "__main__":

  init=dt.now()
  try:
    main()
    end=dt.now()
  except Exception as e:
    print("Error in main:\n",e)
    end = dt.now()
  print('Elapsed time: {}'.format(end-init))

