import numpy as np
import cv2
import blosc2

cv2.ocl.setUseOpenCL(False)


class LazyFrames(object):
    """
    From OpenAI Baseline.
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

    This class provides a solution to optimize the use of memory when
    concatenating different frames, e.g. Atari frames in DQN. The frames are
    individually stored in a list and, when numpy arrays containing them are
    created, the reference to each frame is used instead of a copy.

    """

    def __init__(self, frames, history_length, compress=True):
        self._frames = frames
        self._compress = compress
        if self._compress:
            for s in range(len(self._frames)):
                if isinstance(self._frames[s], np.ndarray):
                    self._frames[s] = (blosc2.compress(self._frames[s]), self._frames[s].shape, self._frames[s].dtype)
        assert len(self._frames) == history_length

    def __array__(self, dtype=None):
        if isinstance(self._frames[0], tuple):
            assert self._compress
            for fi in self._frames:
                assert len(fi) == 3
            shape = self._frames[0][1]
            frames = [np.frombuffer(blosc2.decompress(compressed_data), dtype=dtype)
                      for compressed_data, _, dtype in self._frames]
            for f in frames:
                f.shape = shape
        else:
            frames = self._frames
        out = np.array(frames)
        if dtype is not None:
            out = out.astype(dtype)

        return out

    def copy(self):
        return self

    @property
    def shape(self):
        return (len(self._frames),) + self._frames[0].shape


def preprocess_frame(obs, img_size):
    """
    Convert a frame from rgb to grayscale and resize it.

    Args:
        obs (np.ndarray): array representing an rgb frame;
        img_size (tuple): target size for images.

    Returns:
        The transformed frame as 8 bit integer array.

    """
    image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)

    return np.array(image, dtype=np.uint8)
