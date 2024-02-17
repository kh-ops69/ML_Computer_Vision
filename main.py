# windows users can import keras-gpu to enable working with GPUs which results in much faster execution time
# than using CPU- ensure CUDA libraries are downloaded before hand

import keras.applications
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

print(tf.__version__)
network = keras.applications.VGG19(include_top=False, weights='imagenet')
print(network.summary())

# load image and convert into array(pixel) format, carry out normalization of values, range=(0,1)
content_image = tf.keras.preprocessing.image.load_img('Images/chaves.jpeg')
content_image = tf.keras.preprocessing.image.img_to_array(content_image)
content_image = content_image / 255
content_image = content_image[tf.newaxis, :]

# carry out same process for content and style image separately
style_image = tf.keras.preprocessing.image.load_img('Images/tarsila_amaral.jpg')
style_image = tf.keras.preprocessing.image.img_to_array(style_image)
style_image = style_image / 255
style_image = style_image[tf.newaxis, :]
# print(content_image, "\n", style_image.min(), style_image.max(), style_image.shape)

# select layers from VGG19 network which contain information at different levels for both content and style image
content_layers = ['block4_conv3']
style_layers = ['block1_conv1', 'block2_conv2', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# below function returns a list of layers and their information(trainable params etc.)
def vgg_layers(layer_names):
  vgg = keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]
  network = tf.keras.Model(inputs=[vgg.input], outputs=outputs)

  return network

# get the list of layers from the network and then pass the style image through this network
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image)

# image matrices can be broken down into constituent gram matrices- einsum finds the two matrices that can be multiplied
# to obtain initial image gram matrix
def gram_matrix(layer_activation):
  result = tf.linalg.einsum('bijc,bijd->bcd', layer_activation, layer_activation)
  input_shape = tf.shape(layer_activation)
  num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

  return result / num_locations

# gram_matrix(style_outputs[0])


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super().__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs * 255.0
    # 0 - 1
    # -127.50 - 127.50
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs = outputs[:self.num_style_layers]
    content_outputs = outputs[self.num_style_layers:]

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    # obtain gram matrices and their tf information for both content and style images
    content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
    style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)
results = extractor(content_image)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
# print(style_targets, content_targets)


new_image = tf.Variable(content_image)
# these values determine how much influence content and style images have over the final output respectively
content_weight = 1
style_weight = 100
# loss optimizer ADAM
optimizer = tf.optimizers.Adam(learning_rate=0.05)

epochs = 3000
print_every = 750
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

for epoch in range(epochs):
  with tf.GradientTape() as tape:
    outputs = extractor(new_image)

    content_outputs = outputs['content']
    style_outputs = outputs['style']

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])

    total_loss = content_loss * content_weight / num_content_layers + style_loss * style_weight / num_style_layers

  gradient = tape.gradient(total_loss, new_image)
  optimizer.apply_gradients([(gradient, new_image)])

  new_image.assign(tf.clip_by_value(new_image, 0.0, 1.0))

  if (epoch + 1) % print_every == 0:
    print('Epoch {} | content loss: {} | style loss: {} | total loss {}'.format(epoch + 1, content_loss, style_loss, total_loss))
    plt.imshow(tf.squeeze(new_image, axis=0))
    plt.show()