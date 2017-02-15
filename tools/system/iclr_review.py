import numpy as np
from skimage.io import imsave
import sys
import joblib
sys.path.append('.')
from tools.common import disp_grid
def main():
    img_filename = 'out/images.npz'
    data = joblib.load(img_filename)
    data = np.array(data)
    if len(data.shape) == 5:
        data = data[:, -1]
    elif len(data.shape) == 3:
        data = data[:, None]
    if len(data) == 0:
        return
    data = np.clip(data, 0, 1)
    img = disp_grid(data, border=1, bordercolor=(0.3, 0, .0), normalize=False)
    imsave('out/panel.png', img)

if __name__ == '__main__':
    main()
