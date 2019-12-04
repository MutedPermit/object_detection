import logging
import contextlib2
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util

from create_image_cv import create_dataset

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('root_path', '', 'The root path of the project')
flags.DEFINE_boolean('training', True, 'Define if we creating a training dataset or a validation dataset')
flags.DEFINE_integer('nb_examples', 100, 'Number of images to create')
FLAGS = flags.FLAGS

img_width = 1280
img_height = 1280

labelmap = {
    'square': 1,
    'circle' : 2,
    'triangle': 3,
    'arrow': 4,

}

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = img_height # Image height
  width = img_width # Image width
  filename = example['path'] # Filename of the image. Empty if image is not from file
  with tf.gfile.GFile(filename, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for figure in example['bboxes']:
      xmins += [figure['xmin']]
      xmaxs += [figure['xmax']]
      ymins += [figure['ymin']]
      ymaxs += [figure['ymax']]
      classes_text += [figure['label'].encode('utf8')]
      classes += [ labelmap[figure['label']] ]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  examples = create_dataset(FLAGS.nb_examples, root_path=FLAGS.root_path, train=FLAGS.training)

  num_shards = 10

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_path, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      tf_example = create_tf_example(example)
      shard_idx = idx % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())

  # for example in examples:
  #   tf_example = create_tf_example(example)
  #   writer.write(tf_example.SerializeToString())
  #
  # writer.close()


if __name__ == '__main__':
  tf.app.run()
