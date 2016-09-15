#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/9/13.
'''
from scipy import misc
import os
import time
import tensorflow as tf
import vgg
import model
import reader

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
                           "Path to vgg model weights")
tf.app.flags.DEFINE_string("MODEL_PATH", "models", "Path to read/write trained models")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "train2014", "Path to training images")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2",
                           "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
                           "Which layers to extract style from")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("STYLE_IMAGES", "style.png", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_string("CONTENT_IMAGES_PATH", None, "Path to content image(s)")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Figure out grams and batch sizes! Doesn't make sense ..
def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    # reshape [num_images, -1, num_filters] -1 : infer see the start of p8, 多一个num_images 而已
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size / FLAGS.BATCH_SIZE)

    return grams

def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        size = int(round(FLAGS.IMAGE_SIZE * FLAGS.STYLE_SCALE))
        images = tf.pack([reader.get_image(path, size) for path in style_paths])
        net, _ = vgg.net(FLAGS.VGG_PATH, images)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

def main(argv=None):
    if not FLAGS.CONTENT_IMAGES_PATH:
        print "train a fast nerual style need to set the Content images path"
        return
    content_images = reader.image(
            FLAGS.BATCH_SIZE,
            FLAGS.IMAGE_SIZE,
            FLAGS.CONTENT_IMAGES_PATH,
            epochs=1,
            shuffle=False,
            crop=False)
    generated_images = model.net(content_images / 255.)

    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)
    with tf.Session() as sess:
        file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
        if not file:
            print('Could not find trained model in {0}'.format(FLAGS.MODEL_PATH))
            return
        print('Using model from {}'.format(file))
        saver = tf.train.Saver()
        saver.restore(sess, file)
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        start_time = time.time()
        try:
            while not coord.should_stop():
                print(i)
                images_t = sess.run(output_format)
                elapsed = time.time() - start_time
                start_time = time.time()
                print('Time for one batch: {}'.format(elapsed))

                for raw_image in images_t:
                    i += 1
                    misc.imsave('out{0:04d}.png'.format(i), raw_image)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)



if __name__ == '__main__':
    tf.app.run()