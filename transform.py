import numpy
import torch
import random

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""
  def __call__(self, sample):
    clip_len = len(sample)
    trans_clip = []
    for i in range(clip_len):
        trans_clip.append(torch.from_numpy(sample[i]))
    return trans_clip

class SpacialCrop(object):
  def __init__(self, crop_size=112, crop_position=None):
    self.crop_size = crop_size
    if crop_position is None:
      self.randomize = True
    else:
      self.randomize = False
      self.crop_position = crop_position
      self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

  def __call__(self, sample):
    clip_len = len(sample)
    t, h, w, c = sample[0].shape
    assert h > self.crop_size, "Frame height is smaller than target crop size"
    assert w > self.crop_size, "Frame width is smaller than target crop size"
    
    if self.randomize == True:
      x1 = random.randint(0, w - self.crop_size)
      y1 = random.randint(0, h - self.crop_size)
    elif self.crop_position == 'c':
      x1 = (w - self.crop_size) // 2
      y1 = (h - self.crop_size) // 2
    elif self.crop_position == 'tl':
      x1 = 0
      y1 = 0
    elif self.crop_position == 'tr':
      x1 = w - self.crop_size
      y1 = 0
    elif self.crop_position == 'bl':
      x1 = 0
      y1 = h - self.crop_size
    elif self.crop_position == 'br':
      x1 = w - self.crop_size
      y1 = h - self.crop_size

    x2 = x1 + self.crop_size
    y2 = y1 + self.crop_size

    trans_clip = []
    for i in range(clip_len):
        trans_clip.append(self._crop_clip(sample[i], y1, y2, x1, x2))
    return trans_clip

  def _crop_clip(clip, y1, y2, x1, x2):
      return clip[:,y1:y2,x1:x2,:]