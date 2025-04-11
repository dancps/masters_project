import os
import argparse
import tensorflow as tf


from masters.models.armnet import RMFNet
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
  parser.add_argument('input',  type=str, help='input')
  parser.add_argument('-o', '--output', default='.', type=str, help='output')
  parser.add_argument('-s', '--seed', default=123, type=int, help='output')
  parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
  parser.add_argument('--multichoice', choices=['a', 'b', 'c'], nargs='+', type=str, help='multiple types of arguments. May be called all at the same time.')
  args = parser.parse_args()


  # Parameters
  img_height, img_width = 224, 224
  batch_size = 64

  # Load data
  train_ds = tf.keras.utils.image_dataset_from_directory(
    args.input,
    validation_split=0.2,
    subset="training",
    seed=args.seed,
    image_size=(img_height, img_width),
    batch_size=batch_size).take(10)
  
  val_ds = tf.keras.utils.image_dataset_from_directory(
    args.input,
    validation_split=0.2,
    subset="validation",
    seed=args.seed,
    image_size=(img_height, img_width),
    batch_size=batch_size).take(10)

  # Apply preprocessing to standardize images
  # train_ds = train_ds.map(preprocess_image)
  # val_ds = val_ds.map(preprocess_image)
  
  print(train_ds)
  print(val_ds)

  # Load model
  model = RMFNet() #RNe()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam()

  ###
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), #'categorical_crossentropy', #
    metrics=['accuracy'])
  
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
  )

  print(history)
  # @tf.function
  # def train_step(images, labels):
  #   with tf.GradientTape() as tape:
  #     # training=True is only needed if there are layers with different
  #     # behavior during training versus inference (e.g. Dropout).
  #     predictions = model(images, training=True)
  #     loss = loss_object(labels, predictions)
  #   gradients = tape.gradient(loss, model.trainable_variables)
  #   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  #   train_loss(loss)
  #   train_accuracy(labels, predictions)


  # ###
  # @tf.function
  # def test_step(images, labels):
  #   # training=False is only needed if there are layers with different
  #   # behavior during training versus inference (e.g. Dropout).
  #   predictions = model(images, training=False)
  #   t_loss = loss_object(labels, predictions)

  #   test_loss(t_loss)
  #   test_accuracy(labels, predictions)


  # ###
  # EPOCHS = 5

  # for epoch in range(EPOCHS):
  #   # Reset the metrics at the start of the next epoch
  #   train_loss.reset_state()
  #   train_accuracy.reset_state()
  #   test_loss.reset_state()
  #   test_accuracy.reset_state()

  #   for images, labels in train_ds:
  #     train_step(images, labels)

  #   for test_images, test_labels in test_ds:
  #     test_step(test_images, test_labels)

  #   print(
  #     f'Epoch {epoch + 1}, '
  #     f'Loss: {train_loss.result()}, '
  #     f'Accuracy: {train_accuracy.result() * 100}, '
  #     f'Test Loss: {test_loss.result()}, '
  #     f'Test Accuracy: {test_accuracy.result() * 100}'
  #   )


  


if __name__ == "__main__":
  main()