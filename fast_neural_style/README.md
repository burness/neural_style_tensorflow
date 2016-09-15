# Fast neural style transfer

A blog with the paper [fast neural style](http://arxiv.org/pdf/1603.08155v1.pdf) in [http://hacker.duanshishi.com/?p=1693](http://hacker.duanshishi.com/?p=1693)
In an attempt to learn Tensorflow I've implemented an Image Transformation Network as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) by Johnson et al.

This technique uses loss functions based on a perceptual similarity and style similarity as described by [Gatys et al](http://arxiv.org/abs/1508.06576) to train a transformation network to synthesize the style of one image with the content of arbitrary images. After it's trained for a particular style it can be used to generate stylized images in one forward pass through the transformer network as opposed to 500-2000 forward + backward passes through a pretrained image classification net which is the direct approach.

### Usage

First get the dependecies (COCO training set images and VGG model weights):

`./get_files.sh`

To train a model for fast stylizing:

`python train_fast_neural_style.py --TRAIN_IMAGES_PATH coco_img_path --STYLE_IMAGES style.png --BATCH_SIZE 8`

Where `--TRAIN_IMAGES_PATH` points to a directory of JPEGs to train the model. The paper uses the [COCO image dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) (13GB). With my K20 card I can do a batch_size of 8 images. The paper trains the model for 2 epochs.

To generate images fast with an already trained model:

`python gen_image.py --CONTENT_IMAGES path_to_images_to_transform`

### Requirements

 - python 2.7.x
 - [Tensorflow r0.10](https://www.tensorflow.org/)
 - [VGG-19 model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
 - [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

### Acknowledgement

- [fast-neural-style with tensorflow](https://github.com/OlavHN/fast-neural-style)
- [Chainer implementation] (https://github.com/yusuketomoto/chainer-fast-neuralstyle)
- [Tensorflow Neural style implementation] (https://github.com/anishathalye/neural-style) (Both inspiration and copied the VGG code)
