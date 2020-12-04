"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import csv
start_time = time.time()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import logging
import tensorflow as tf
logging.getLogger('tensorflow').disabled = True

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

slim = tf.contrib.slim


tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', '/home/lms/TI_FGSM/checkpoint/inception_v3.ckpt', 'Path to checkpoint for inception network.')

# tf.flags.DEFINE_string('checkpoint_path_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens3_adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', '/home/lms/TI_FGSM/checkpoint/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2', "/home/lms/TI_FGSM/checkpoint/inception_resnet_v2_2016_08_30.ckpt", 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', "/home/lms/TI_FGSM/checkpoint/resnet_v2_152.ckpt", 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet_50', "/home/lms/TI_FGSM/checkpoint/resnet_v2_50.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet_101', "/home/lms/TI_FGSM/checkpoint/resnet_v2_101.ckpt",
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens4_adv_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens_adv_inception_resnet_v2', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_adv_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_ens3_adv_inception_v3', '/data/ltj/Pycharm_Projects/Po-Attack/checkpoint/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_dir', "/home/lms/TI_FGSM/data/", 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', "/data/ltj/Pycharm_Projects/Po-Attack/result/", 'Output directory with images.')

tf.flags.DEFINE_string('label_csv', "/home/lms/TI_FGSM/dev_dataset.csv", 'label information with csv file.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 5, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 50, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 0.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_string('logname', '16-20-ince_res_v2.txt', 'name of log file')

tf.flags.DEFINE_string('cuda', '3', "which cuda device can use")

tf.flags.DEFINE_integer('top_k', 1, "top_k")

tf.flags.DEFINE_string('eval_y', 'y_tar', 'test y_true or y_tar')

tf.flags.DEFINE_string('att_model', 'incep_v3', 'which model is black-box attack')

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


kernel = gkern(15, 3).astype(np.float32)
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






def Evaluator(x, y):
    num_classes = 1001

    # should keep original x here for output

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            x, num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            x, num_classes=num_classes, is_training=False)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_50, end_points_resnet_50 = resnet_v2.resnet_v2_50(
            x, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     logits_ens_v3, end_points_ens_v3 = inception_v3.inception_v3(
    #         x, num_classes=num_classes, is_training=False)

    # acc_v3 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_v3['Predictions'], 1), y), tf.float32))
    # acc_v4 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_v4['Predictions'], 1), y), tf.float32))
    # acc_res_v2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_res_v2['Predictions'], 1), y), tf.float32))
    # acc_resnet = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_resnet['predictions'], 1), y), tf.float32))
    # acc_resnet_50 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_resnet_50['predictions'], 1), y), tf.float32))
    # acc_resnet_101 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_resnet_101['predictions'], 1), y), tf.float32))
    # acc_ens_v3 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_ens_v3['predictions'], 1), y), tf.float32))

    logits_esm = (logits_v3 + logits_v4 + logits_res_v2 + logits_resnet + logits_resnet_50 + logits_resnet_101)
    if FLAGS.att_model == "incep_v3":
        logits_esm = (logits_esm - logits_v3) / 5
    elif FLAGS.att_model == "incep_v4":
        logits_esm = (logits_esm - logits_v4) / 5
    elif FLAGS.att_model == "incep_res_v2":
        logits_esm = (logits_esm - logits_res_v2) / 5
    elif FLAGS.att_model == "resnet_50":
        logits_esm = (logits_esm - logits_resnet_50) / 5
    elif FLAGS.att_model == "resnet_101":
        logits_esm = (logits_esm - logits_resnet_101) / 5
    elif FLAGS.att_model == "resnet_152":
        logits_esm = (logits_esm - logits_resnet) / 5
    elif FLAGS.att_model == "ens3_adv_3":
        logits_esm = (logits_esm + logits_ens4_adv_v3 + logits_ensadv_res_v2) / 8
    elif FLAGS.att_model == "ens4_adv_3":
        logits_esm = (logits_esm + logits_ens3_adv_v3 + logits_ensadv_res_v2) / 8
    elif FLAGS.att_model == "ensadv_res_2":
        logits_esm = (logits_esm + logits_ens4_adv_v3 + logits_ens3_adv_v3) / 8
    # top_k
    top_k = FLAGS.top_k
    acc_v3 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_v3['Predictions'], y, k=top_k), tf.float32))
    acc_v4 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_v4['Predictions'], y, k=top_k), tf.float32))
    acc_res_v2 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_res_v2['Predictions'], y, k=top_k), tf.float32))
    acc_resnet = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_resnet['predictions'], y, k=top_k), tf.float32))
    acc_resnet_50 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_resnet_50['predictions'], y, k=top_k), tf.float32))
    acc_resnet_101 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_resnet_101['predictions'], y, k=top_k), tf.float32))
    acc_adv_v3 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_adv_v3['Predictions'], y, k=top_k), tf.float32))
    acc_ens3_adv_v3 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_ens3_adv_v3['Predictions'], y, k=top_k), tf.float32))
    acc_ens4_adv_v3 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_ens4_adv_v3['Predictions'], y, k=top_k), tf.float32))
    acc_ensadv_res_v2 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(end_points_ensadv_res_v2['Predictions'], y, k=top_k), tf.float32))

    acc_esm = tf.reduce_sum(tf.cast(tf.nn.in_top_k(slim.softmax(logits_esm, scope='predictions'), y, k=top_k), tf.float32))
    # acc_ens_v3 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(end_points_ens_v3['predictions'], 1), y), tf.float32))

    # pred1 = tf.argmax(end_points_v3['Predictions'], 1)
    # pred2 = tf.argmax(end_points_v4['Predictions'], 1)
    # pred3 = tf.argmax(end_points_res_v2['Predictions'], 1)
    # pred4 = tf.argmax(end_points_resnet['predictions'], 1)

    return acc_v3, acc_v4, acc_res_v2, acc_resnet, acc_resnet_50, acc_resnet_101, acc_adv_v3, acc_ens3_adv_v3, \
           acc_ens4_adv_v3, acc_ensadv_res_v2, acc_esm, end_points_v3, end_points_v4, end_points_res_v2, end_points_resnet





def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    # eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    print(time.time() - start_time)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)


        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        # predicted_labels = tf.argmax(end_points['Predictions'], 1)
        # y = tf.one_hot(predicted_labels, num_classes)
        true_label_list, target_label_list = label_dict(FLAGS.label_csv)
        y_true = tf.placeholder(tf.int64, shape=batch_shape[0])

        acc_v3, acc_v4, acc_res_v2, acc_resnet, acc_resnet_50, acc_resnet_101, acc_adv_v3, acc_ens3_adv_v3, \
        acc_ens4_adv_v3, acc_ensadv_res_v2, acc_esm, pred1, pred2, pred3, pred4 = Evaluator(x_input, y_true)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s11 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s13 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        acc_num_v3_ = 0
        acc_num_v4_ = 0
        acc_num_res_v2_ = 0
        acc_num_resnet_ = 0
        acc_num_resnet_50 = 0
        acc_num_resnet_101 = 0
        acc_num_adv_v3_ = 0
        acc_num_ens3_adv_v3_ = 0
        acc_num_ens4_adv_v3_ = 0
        acc_num_ensadv_res_v2_ = 0
        acc_num_esm_ = 0

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)

            s7.restore(sess, FLAGS.checkpoint_path_resnet_50)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)
            s9.restore(sess, FLAGS.checkpoint_path_resnet_101)
            s10.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s11.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s12.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
            s13.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            print(time.time() - start_time)
            num = 0

            for filenames, images, y, y_tar in load_images(FLAGS.output_dir, batch_shape,true_label_list, target_label_list):
                # y = tf.one_hot(y, num_classes)
                eval_y = y_tar
                if FLAGS.eval_y == 'y_true':
                    eval_y = y
                acc_v3_, acc_v4_, acc_res_v2_, acc_resnet_,acc_resnet_50_,acc_resnet_101_,acc_adv_v3_,acc_ens3_adv_v3_, acc_ens4_adv_v3_, acc_ensadv_res_v2_, acc_esm_, pred1_, pred2_, pred3_, pred4_ = \
                    sess.run([acc_v3, acc_v4, acc_res_v2, acc_resnet, acc_resnet_50, acc_resnet_101, acc_adv_v3, acc_ens3_adv_v3, acc_ens4_adv_v3, acc_ensadv_res_v2, acc_esm, pred1, pred2, pred3, pred4],
                             feed_dict={x_input: images, y_true: eval_y})
                num += batch_shape[0]

                acc_num_v3_ = acc_num_v3_+acc_v3_
                acc_num_v4_ = acc_num_v4_ + acc_v4_
                acc_num_res_v2_ = acc_num_res_v2_ + acc_res_v2_
                acc_num_resnet_ = acc_num_resnet_ + acc_resnet_
                acc_num_resnet_50 = acc_num_resnet_50 +  acc_resnet_50_
                acc_num_resnet_101 = acc_num_resnet_101 + acc_resnet_101_
                acc_num_adv_v3_ = acc_num_adv_v3_ + acc_adv_v3_
                acc_num_ens3_adv_v3_ = acc_num_ens3_adv_v3_ + acc_ens3_adv_v3_
                acc_num_ens4_adv_v3_ = acc_num_ens4_adv_v3_ + acc_ens4_adv_v3_
                acc_num_ensadv_res_v2_ = acc_num_ensadv_res_v2_ + acc_ensadv_res_v2_
                acc_num_esm_ = acc_num_esm_ + acc_esm_

        print(acc_num_v3_, acc_num_v4_, acc_num_res_v2_, acc_num_resnet_, acc_num_resnet_50, acc_num_resnet_101, acc_num_adv_v3_, acc_num_ens3_adv_v3_, acc_num_ens4_adv_v3_, acc_num_ensadv_res_v2_, acc_num_esm_, num)
        log = [str(acc_num_v3_)+"\n", str(acc_num_v4_)+"\n", str(acc_num_res_v2_)+"\n", str(acc_num_resnet_)+"\n", str(acc_num_resnet_50)+"\n",
               str(acc_num_resnet_101)+"\n", str(acc_num_adv_v3_)+"\n", str(acc_num_ens3_adv_v3_)+"\n", str(acc_num_ens4_adv_v3_)+"\n", str(acc_num_ensadv_res_v2_)+"\n", str(acc_num_esm_)+"\n"]
        with open(FLAGS.logname, "w") as f:
            f.writelines(log)
        # print(acc_num_v3_, acc_num_v4_, acc_num_res_v2_, acc_num_resnet_)

        print(time.time() - start_time)


if __name__ == '__main__':
    tf.app.run()
