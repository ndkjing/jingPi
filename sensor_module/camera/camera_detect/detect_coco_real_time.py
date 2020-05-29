from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

from PIL import Image
from PIL import ImageDraw


def _round_up(value, n):
    """Rounds up the given value to the next number divisible by n.
    Args:
      value: int to be rounded up.
      n: the number that should be divisible into value.
    Returns:
      the result of value rounded up to the next multiple of n.
    """
    return n * ((value + (n - 1)) // n)


def _round_buffer_dims(dims):
    """Appropriately rounds the given dimensions for image overlaying.
    As per the PiCamera.add_overlay documentation, the source data must have a
    width rounded up to the nearest multiple of 32, and the height rounded up to
    the nearest multiple of 16. This does that for the given image dimensions.
    Args:
      dims: image dimensions.
    Returns:
      the rounded-up dimensions in a tuple.
    """
    width, height = dims
    return _round_up(width, 32), _round_up(height, 16)


class Annotator:
    """Utility for managing annotations on the camera preview."""

    def __init__(self, camera, default_color=None):
        """Initializes Annotator parameters.
        Args:
          camera: picamera.PiCamera camera object to overlay on top of.
          default_color: PIL.ImageColor (with alpha) default for the drawn content.
        """
        self._camera = camera
        self._dims = camera.resolution
        self._buffer_dims = _round_buffer_dims(self._dims)
        self._buffer = Image.new('RGBA', self._buffer_dims)
        self._overlay = None
        self._draw = ImageDraw.Draw(self._buffer)
        self._default_color = default_color or (0xFF, 0, 0, 0xFF)

    def update(self):
        """Draws any changes to the image buffer onto the overlay."""
        # For some reason, simply updating the current overlay causes
        # PiCameraMMALError every time we update. To avoid that, we create a new
        # overlay each time we want to update.
        # We use a temp overlay object because if we remove the current overlay
        # first, it causes flickering (the overlay visibly disappears for a moment).
        temp_overlay = self._camera.add_overlay(
            self._buffer.tobytes(), format='rgba', layer=3, size=self._buffer_dims)
        if self._overlay is not None:
            self._camera.remove_overlay(self._overlay)
        self._overlay = temp_overlay
        self._overlay.update(self._buffer.tobytes())

    def clear(self):
        """Clears the contents of the overlay, leaving only the plain background."""
        self._draw.rectangle((0, 0) + self._dims, fill=(0, 0, 0, 0x00))

    def bounding_box(self, rect, outline=None, fill=None):
        """Draws a bounding box around the specified rectangle.
        Args:
          rect: (x1, y1, x2, y2) rectangle to be drawn, where (x1, y1) and (x2, y2)
            are opposite corners of the desired rectangle.
          outline: PIL.ImageColor with which to draw the outline (defaults to the
            Annotator default_color).
          fill: PIL.ImageColor with which to fill the rectangle (defaults to None,
            which will *not* cover up drawings under the region).
        """
        outline = outline or self._default_color
        self._draw.rectangle(rect, fill=fill, outline=outline)

    def text(self, location, text, color=None):
        """Draws the given text at the given location.
        Args:
          location: (x, y) point at which to draw the text (upper left corner).
          text: string to be drawn.
          color: PIL.ImageColor to draw the string in (defaults to the Annotator
            default_color).
        """
        color = color or self._default_color
        self._draw.text(location, text, fill=color)


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(annotator, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        annotator.bounding_box([xmin, ymin, xmax, ymax])
        annotator.text([xmin, ymin],
                       '%s\n%.2f' % (labels[obj['class_id']], obj['score']))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', default='detect.tflite')
    parser.add_argument(
        '--labels', help='File path of labels file.', default='labelmap.txt')
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            annotator = Annotator(camera)
            for _ in camera.capture_continuous(
                    stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize(
                    (input_width, input_height), Image.ANTIALIAS)
                start_time = time.monotonic()
                results = detect_objects(interpreter, image, args.threshold)
                elapsed_ms = (time.monotonic() - start_time) * 1000

                annotator.clear()
                annotate_objects(annotator, results, labels)
                annotator.text([5, 0], '%.1fms' % (elapsed_ms))
                annotator.update()

                stream.seek(0)
                stream.truncate()

        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()
