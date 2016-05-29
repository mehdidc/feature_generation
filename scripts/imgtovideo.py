from tempfile import mkdtemp
import shutil
import os
import sys
import subprocess

from skimage.io import imsave

cmd_tpl = 'ffmpeg -framerate {framerate} -i {pattern} -c:v libx264 -r {rate} -pix_fmt yuv420p {out}'


def imgs_to_video(imgs, out='out.mp4', framerate=20, rate=20, verbose=0):
    """
    Utility function to convert a set of images to a video.
    images are represented by a 3D tensor of shape (nbimages, w, h) or
    (nbimages, w, h, 3) for colored images.

    Assumes 'ffmpeg' is installed.

    imgs : 3D tensor of shape (nbimages, w, h) or (nbimages, w, h, 3)
    out : output file to write
    framerate : framerate of input (ffmpeg option)
    rate : rate (ffmpeg option)
    verbose : turn to 1 to see the details
    """
    out = os.path.abspath(out)
    dirname = mkdtemp(prefix='img_to_video')
    for i, img in enumerate(imgs):
        filename = os.path.join(dirname, 'img{:08d}.png'.format(i))
        imsave(filename, img)
    params = dict(
        framerate=framerate,
        pattern='img%08d.png',
        rate=rate,
        out=out
    )
    cmd = cmd_tpl.format(**params)
    if verbose > 0:
        stdout = sys.stdout
        print(cmd)
    else:
        stdout = open(os.devnull, 'w')
    subprocess.call(cmd, shell=True, cwd=dirname, stdout=stdout, stderr=stdout)
    shutil.rmtree(dirname)

if __name__ == '__main__':
    import numpy as np
    imgs = np.random.uniform(size=(100, 12, 12, 3))
    imgs_to_video(imgs, out='out.mp4')