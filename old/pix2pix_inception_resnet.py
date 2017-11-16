from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import warnings
from functools import partial
from nets import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope



parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--monochrome", dest="monochrome", action="store_true", help="convert image from rgb to gray")
parser.set_defaults(monochrome=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

#YuhangLi
parser.add_argument("--num_unet", type=int, default=10, help="number of u-connection layers")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def blockA(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def blockB(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def blockC(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             # output_stride=16,
                             align_feature_maps=False,
                             scope=None):
  """Inception model from  http://arxiv.org/abs/1602.07261.

  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  padding = 'SAME' 

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, padding=padding, scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding, scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3,  padding=padding, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      
      # 73 x 73 x 64
      # net = slim.max_pool2d(net, 3, stride=2, padding=padding,
      #                      scope='MaxPool_3a_3x3')
      #if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      
      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding, scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding, scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      
      # 35 x 35 x 192
      # net = slim.max_pool2d(net, 3, stride=2, padding=padding,
      #                      scope='MaxPool_5a_3x3')
      # if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points

      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

      if add_and_check_final('Mixed_5b', net): return net, end_points
      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17)

      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = True

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

      if add_and_check_final('Mixed_6a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10)
      if add_and_check_final('PreAuxLogits', net): return net, end_points

      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

      if add_and_check_final('Mixed_7a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

  

def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_v2_base(inputs, scope=scope)

      if create_aux_logits:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool_1a_8x8')
        net = slim.flatten(net)

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')

        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
inception_resnet_v2.default_image_size = 299


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Returns the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope

        
def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    
    with tf.name_scope("load_images"):
        # Use a queue to save input file names
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        # reader, only read file and get paths(tensor) and content(tensor, not image tensor)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        # decode contents into image tensor
        raw_input = decode(contents)
        # [0, 255] to [0.0, 1.0]
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        #assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        #with tf.control_dependencies([assertion]):
            #raw_input = tf.identity(raw_input)
        
        # YuhangLi: if four channels (rgba), get rgb only 
        raw_input = tf.identity(raw_input[:, :, 0:3])
        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            
            # [0.0, 1.0] to [-1.0, 1.0]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])
            
           
    
    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
        
        if a.monochrome:
            r = tf.image.rgb_to_grayscale(r)
            
        if 0: #height != width:
            # crop to correct ratio
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            dst = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
            
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")    
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))



    
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return output
    
    
def create_generator(generator_inputs, generator_outputs_channels):
    with tf.variable_scope('Generator'):
        shortcut = slim.conv2d(generator_inputs, a.ngf, [1, 1], stride=1,
                             normalizer_fn=None, activation_fn=tf.nn.relu)
        # seperate one bottleneck operation in case of the scope problem
        residual = bottleneck(shortcut , depth=64, depth_bottleneck=64, stride=1)                  
        residual = slim.repeat(residual, 10, bottleneck, depth=64, depth_bottleneck=64, stride=1)
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = shortcut + residual
        
        net = slim.conv2d(residual, a.ngf, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = slim.conv2d(residual, a.ngf, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        output = slim.conv2d(residual, 3, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None)                             
    return output

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]
        
    def create_discriminator_global(discrim_inputs, discrim_targets):
        n_layers = 6
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1_global"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
        # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 16]
        # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 32]
        # layer_7: [batch, 4, 4, ndf * 32] => [batch, 2,  2,  ndf * 64]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
                out_channels = a.ndf * 2**(i+1)
                stride =  2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_8: [batch, 2, 2, ndf * 64] => [batch, 1, 1, 1]
        with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator_patch"):
        with tf.variable_scope("discriminator_patch"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_patch = create_discriminator(inputs, targets)
    
    with tf.name_scope("real_discriminator_global"):
        with tf.variable_scope("discriminator_global"):
            # 2x [batch, height, width, channels] => [batch, 1, 1, 1]
            predict_real_global = create_discriminator_global(inputs, targets)
    
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator_patch", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)
            
    with tf.name_scope("fake_discriminator_global"):
        with tf.variable_scope("discriminator_global", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 1, 1, 1]
            predict_fake_global = create_discriminator_global(inputs, outputs)            

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real_patch + EPS) \
            + tf.log(predict_real_global + EPS) \
            + tf.log(1 - predict_fake + EPS)))
    
   
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train_patch"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
     
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real_patch,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
