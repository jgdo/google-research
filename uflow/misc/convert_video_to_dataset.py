# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Parses a video into a tf.data.dataset of TFRecords."""

import os

from absl import app
from absl import flags
import cv2
import tensorflow as tf
import itertools

from uflow.data_conversion_scripts import conversion_utils

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('video_path', '', 'Location of the mp4 video file.')
flags.DEFINE_string('output_path', '', 'Location to write the video dataset.')
flags.DEFINE_integer('frame_skip', 0, 'Frame skip in dataset.')

def write_data_example(record_writer, image1, image2):
  """Write data example to disk."""
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  feature = {
      'height': conversion_utils.int64_feature(image1.shape[0]),
      'width': conversion_utils.int64_feature(image1.shape[1]),
  }
  example = tf.train.SequenceExample(
      context=tf.train.Features(feature=feature),
      feature_lists=tf.train.FeatureLists(
          feature_list={
              'images':
                  tf.train.FeatureList(feature=[
                      conversion_utils.bytes_feature(
                          image1.astype('uint8').tobytes()),
                      conversion_utils.bytes_feature(
                          image2.astype('uint8').tobytes())
                  ]),
          }))
  record_writer.write(example.SerializeToString())


def convert_video(video_file_path_list, output_folder, frame_skip):
  """Converts video at video_file_path to a tf.data.dataset at output_folder."""
  frame_skip = max(frame_skip, 0)
  if not tf.io.gfile.exists(output_folder):
    print('Making new plot directory', output_folder)
    tf.io.gfile.makedirs(output_folder)
  filename = os.path.join(output_folder, 'fvideo@1')
  with tf.io.TFRecordWriter(filename) as record_writer:
      count = 0
      for video_file_path in video_file_path_list:
        vidcap = cv2.VideoCapture(video_file_path)
        success, image1 = vidcap.read()
        assert success, "Could not read video file: {}".format(video_file_path)
        for i in itertools.count(start=1, step=1):
          success, image2 = vidcap.read()
          if not success:
            break
          if i % (frame_skip+1) != 0:
              tf.compat.v1.logging.info('Skipping frame')
              continue
          tf.compat.v1.logging.info('Read a new frame: %d', count)
          write_data_example(record_writer, image1, image2)
          image1 = image2
          count += 1
        vidcap.release()

def main(unused_argv):
  print(FLAGS.video_path)
  convert_video(FLAGS.video_path, FLAGS.output_path, FLAGS.frame_skip)

if __name__ == '__main__':
  app.run(main)
