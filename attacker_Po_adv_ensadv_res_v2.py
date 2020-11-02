"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import csv

start_time = time.time()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import logging

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# logging.getLogger('tensorflow').disabled = True

slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', '/home/lms/TI_FGSM/checkpoint/inception_v3.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', '/home/lms/TI_FGSM/checkpoint/inception_v4.ckpt',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2',
                       "/home/lms/TI_FGSM/checkpoint/inception_resnet_v2_2016_08_30.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', "/home/lms/TI_FGSM/checkpoint/resnet_v2_152.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet_50', "/home/lms/TI_FGSM/checkpoint/resnet_v2_50.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet_101', "/home/lms/TI_FGSM/checkpoint/resnet_v2_101.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens4_adv_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens_adv_inception_resnet_v2', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens3_adv_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_dir', "/home/lms/TI_FGSM/data/", 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', "/data/ltj/Pycharm_Projects/Po-Attack/result/", 'Output directory with images.')

tf.flags.DEFINE_string('label_csv', "/home/lms/TI_FGSM/dev_dataset.csv", 'label information with csv file.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 5, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_float('W_crs', 0.0002, "weight of cross-entry loss")

tf.flags.DEFINE_string('loss', 'po', 'ce: cross_entry, po: poincare, cepo: ce+po')

tf.flags.DEFINE_string('cuda', '3', 'which cuda device can use')

tf.flags.DEFINE_integer('kern', 5, 'kernel len of function gkern')

tf.flags.DEFINE_float('W_aux', 0.4, 'weight of loss in auxlogits and y_target')

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

kernel = gkern(FLAGS.kern, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape, dict_true, dict_target):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    y_true = np.zeros(batch_shape[0])
    y_target = np.zeros(batch_shape[0])
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath,'rb') as f:
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0

        filename = os.path.basename(filepath)
        filenames.append(filename)
        y_true[idx] = dict_true[filename]
        y_target[idx] = dict_target[filename]
        idx += 1
        if idx == batch_size:
            yield filenames, images, y_true, y_target
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, y_true, y_target

def label_dict(filename):
    _true_labels = {}
    _target_labels = {}
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        try:
            row_idx_image_id = header_row.index('ImageId')
            row_idx_true_label = header_row.index('TrueLabel')
            row_idx_target_label = header_row.index('TargetClass')
        except ValueError:
            raise IOError('Invalid format of dataset metadata.')
        for row in reader:
            if len(row) < len(header_row):
                # skip partial or empty lines
                continue
            try:
                image_id = row[row_idx_image_id]+'.png'
                _true_labels[image_id] = int(row[row_idx_true_label])
                _target_labels[image_id] = int(row[row_idx_target_label])
            except (IndexError, ValueError):
                raise IOError('Invalid format of dataset metadata')
    return _true_labels, _target_labels

def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')
def Poincare_dis(a,b):
    L2_a = tf.reduce_sum(tf.square(a), 1)
    L2_b = tf.reduce_sum(tf.square(b), 1)

    theta = 2 * tf.reduce_sum(tf.square(a - b), 1) / (
            (1 - L2_a) * (1 - L2_b))
    distance = tf.reduce_mean(tf.acosh(1.0 + theta))
    return distance

def Cos_dis(a, b):
    a_b = tf.abs(tf.reduce_sum(tf.multiply(a, b), 1))
    L2_a = tf.reduce_sum(tf.square(a), 1)
    L2_b = tf.reduce_sum(tf.square(b), 1)
    distance = a_b / tf.sqrt(L2_a * L2_b)
    return distance

def graph(x, y, i, x_max, x_min, grad, y_target, y_logits):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    # should keep original x here for output

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_50, end_points_resnet_50 = resnet_v2.resnet_v2_50(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
            input_diversity(x),  num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    # Adv training models
    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
    #         input_diversity(x), num_classes=num_classes, is_training=False, scope='AdvInceptionV3')
    #
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

    # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    #     logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
    #         input_diversity(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

    logits = (logits_v4 + logits_res_v2 + logits_v3 + logits_resnet + logits_resnet_50 + logits_resnet_101 + logits_ens3_adv_v3 + logits_ens4_adv_v3) / 8

    auxlogits = (end_points_v4['AuxLogits'] + end_points_res_v2['AuxLogits'] + end_points_v3['AuxLogits'] +
                 end_points_ens3_adv_v3['AuxLogits'] + end_points_ens4_adv_v3['AuxLogits']) / 5
    
    y_oh = tf.one_hot(y, num_classes)
    y_target_oh = tf.one_hot(y_target, num_classes)


    loss = -FLAGS.W_crs * tf.losses.softmax_cross_entropy(y_oh,
                                            logits,
                                            label_smoothing=0.0,
                                            weights=1.0)
    # loss = - Poincare_dis(tf.clip_by_value((y_oh-0.01), 0.0, 1.0),
    #                              logits / tf.reduce_sum(tf.abs(logits), [1], keep_dims=True) )



    loss_ce = tf.losses.softmax_cross_entropy(y_target_oh,
                                              logits,
                                              label_smoothing=0.0,
                                              weights=1.0)

    loss_ce += tf.losses.softmax_cross_entropy(y_target_oh,
                                               auxlogits,
                                               label_smoothing=0.0,
                                               weights=0.9)

    loss_po = Poincare_dis(tf.clip_by_value((y_target_oh-0.00001), 0.0, 1.0),
                                 logits / tf.reduce_sum(tf.abs(logits), [1], keep_dims=True))

    loss_po += FLAGS.W_aux*Poincare_dis(tf.clip_by_value((y_target_oh-0.00001), 0.0, 1.0),
                                auxlogits / tf.reduce_sum(tf.abs(auxlogits), [1], keep_dims=True))

    loss_cos = tf.clip_by_value((Cos_dis(y_oh, logits) - Cos_dis(y_target_oh, logits) + 0.007), 0.0, 2.1)

    if FLAGS.loss == "ce":
        loss = loss_ce
    elif FLAGS.loss == "po":
        loss = loss_po
    elif FLAGS.loss == "trip_po":
        loss = loss_po + 0.01*loss_cos
    # loss += cross_entropy
    # loss += -10*Cos_dis(y_target_oh, logits)

    noise = -tf.gradients(loss, x)[0]
    # TI-
    # noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    # CE  Cross-entry loss must add this term
    if FLAGS.loss == "ce":
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)

    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise,  y_target, logits



def stop(x, y, i, x_max, x_min, grad, y_target, y_logits):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    print(time.time() - start_time)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        # with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        #     _, end_points = inception_resnet_v2.inception_resnet_v2(
        #         x_input, num_classes=num_classes, is_training=False)
        true_label_list,target_label_list = label_dict(FLAGS.label_csv)

        # predicted_labels = tf.argmax(end_points['Predictions'], 1)
        # y = tf.one_hot(predicted_labels, num_classes)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        y_true = tf.placeholder(tf.int64, shape=batch_shape[0])
        y_target = tf.placeholder(tf.int64, shape=batch_shape[0])
        y_logits = tf.placeholder(tf.float32, shape=[batch_shape[0], num_classes])
        temp = np.ones([batch_shape[0], num_classes]).astype(np.float32)
        x_adv, _, _, _, _, noise_, _, logits_ = tf.while_loop(stop, graph, [x_input, y_true, i, x_max, x_min, grad, y_target,  y_logits])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s11 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            s7.restore(sess, FLAGS.checkpoint_path_resnet_50)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)
            s9.restore(sess, FLAGS.checkpoint_path_resnet_101)
            s10.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            # s11.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s12.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            print(time.time() - start_time)

            for filenames, images, y, y_tar in load_images(FLAGS.input_dir, batch_shape,true_label_list,target_label_list):
                # y = tf.one_hot(y, num_classes)
                # y_tar = tf.one_hot(y_tar, num_classes)
                adv_images, grad, logits_c = sess.run([x_adv, noise_, logits_], feed_dict={x_input: images, y_true: y, y_target:y_tar, y_logits: temp})
                save_images(adv_images, filenames, FLAGS.output_dir)

        print(time.time() - start_time)


if __name__ == '__main__':
    tf.app.run()
