import tensorflow as tf

def imaloader(iurl, inputer):
  images = tf.io.decode_image(iurl, channels=3)
  images = tf.image.resize(images, inputer)
  images = tf.expand_dims(images, axis=0)
  result = tf.cast(images / 255.0, tf.float32)
  return result