#!/usr/bin/env python

import sys
from PyQt4.QtCore import *  # NOQA
from PyQt4.QtGui import *  # NOQA
from collections import defaultdict
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from lasagne import layers as L


def qimage_from_array_(data):
    w, h = data.shape[0], data.shape[1]
    image = QImage(QSize(w, h), QImage.Format_RGB32)
    for i in range(w):
        for j in range(h):
            val = int(data[j, i] * 255)
            c = qRgb(val, val, val)
            image.setPixel(i, j, c)
    return image


def qimage_to_array_(qimage):
    w, h = qimage.size().width(), qimage.size().height()
    data = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            c = qimage.pixel(i, j)
            c = QColor.fromRgb(c).red()
            data[j, i] = c
    return data


def is_conv_layer(name):
    return name.startswith("conv") or name.startswith("unconv") or name.startswith("wta")


class Window(QWidget):

    def __init__(self, capsule, data, w, h, c, max_feature_maps=np.inf):
        super(Window, self).__init__()

        self.capsule = capsule

        self.w = w
        self.h = h
        self.data = data
        self.max_feature_maps = max_feature_maps

        self.resize(600, 600)
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        empty = QImage(QSize(w, h), QImage.Format_RGB32)
        empty.fill(QColor('white'))

        self.painters = defaultdict(list)
        layers = capsule.layers
        conv_layer_names = ["input"]
        conv_layer_names += filter(is_conv_layer, layers.keys())
        conv_layer_names += ["output"]
        self.conv_layer_names = conv_layer_names

        get_output = OrderedDict()
        x = T.tensor4()
        for i in range(len(self.conv_layer_names)):
            cur_name = self.conv_layer_names[i]
            cur = layers[cur_name]
            next_layers = self.conv_layer_names[i + 1:]
            if len(next_layers) == 0:
                break
            o = []
            for l in next_layers:
                o.append(L.get_output(layers[l], {cur: x}))
            f = theano.function([x], o)
            get_output[cur_name] = f
        self.get_output = get_output

        j = 0
        for name in self.conv_layer_names:
            layer = capsule.layers[name]
            w, h = layer.output_shape[2:]
            nb = layer.output_shape[1]
            nb = min(nb, self.max_feature_maps)

            if name == "input":
                X = data.X[0]
                X = X.reshape((w, h))
                image = qimage_from_array_(X)
            else:
                image = qimage_from_array_(np.zeros((w, h)))
            for i in range(layer.output_shape[1]):
                painter = ImagePainter(image)
                painter.window = self
                painter.layer_name = name
                self.painters[name].append(painter)
                if i < nb:
                    self.grid.addWidget(self.painters[name][-1], i, j)
            j += 1
        self.apply_button = QPushButton("Random!")
        self.apply_button.clicked.connect(self.random_event)
        self.grid.addWidget(self.apply_button, 1, 0)
        self.cur_sort = {}

    def random_event(self):
        self.data.load()
        X = self.data.X[0]
        if len(X.shape) > 2:
            X = X[0]
        X = (X.reshape((self.w, self.h)))
        image = qimage_from_array_(X)
        self.painters["input"][0].set_image(image)
        self.repaint()
        self.apply_event("input")

    def apply_event(self, layer_name):
        idx = self.conv_layer_names.index(layer_name)
        next_layers = self.conv_layer_names[idx + 1:]
        images = [painter.get_image() 
                  for painter in self.painters[layer_name]]
        images = map(qimage_to_array_, images)
        images = np.array(images)
        if "unconv" in layer_name or "wta" in layer_name:
            power = 1
        else:
            power = 1
        images = (images / 255.) * power
        images = images.astype(np.float32)

        if layer_name in self.cur_sort:
            s = self.cur_sort[layer_name][0]
            s_rev = [None] * len(s)
            for i, a in enumerate(s):
                s_rev[a] = i
            s_rev = np.array(s_rev)
            images = images[s_rev]
        else:
            print(layer_name)

        b, w, h = self.capsule.layers[layer_name].output_shape[1:]
        a = 1
        images = images.reshape((a, b, w, h))
        capsule = self.capsule

        if layer_name == "input":
            images = capsule.preprocess(images)
        results = self.get_output[layer_name](images)
        
        S = {}
        for name, result in zip(next_layers, results):
            nb = min(result.shape[1], self.max_feature_maps)
            result_norm = (np.abs(result)).sum(axis=(2, 3))
            result_sorted = np.argsort(-result_norm, axis=1)
            S[name] = result_sorted
            for i, s in enumerate(result_sorted):
                result[i] = result[i, s]
            result = result[:, 0:nb]
            result = result.reshape(
                (result.shape[0] * result.shape[1],
                 result.shape[2], result.shape[3]))
            for i in range(len(result)):
                r = result[i]
                # if name == "output":
                #    import matplotlib.pyplot as plt
                #    plt.imshow(1-r, cmap='gray', interpolation='none')
                #    plt.show()
                r -= r.min()
                if r.max() != 0:
                    r /= (r.max())
                qimage = qimage_from_array_(r)
                self.painters[name][i].set_image(qimage)
        self.cur_sort.update(S)
        F = self.capsule.layers["unconv"].W.get_value()
        F = F[0]
        name = self.conv_layer_names[-3]
        if name in S:
            s = S[name][0]
            nb = min(s.shape, self.max_feature_maps)
            s = s[0:nb]
            F = F[s]
            from lasagnekit.misc.plot_weights import tile_raster_images
            import matplotlib.pyplot as plt
            #F = F[:, ::-1, ::-1]
            img = tile_raster_images(F, F.shape[1:], (nb, 1))
            plt.imshow(img, cmap="gray")
            plt.axis('off')
            plt.show()


class ImagePainter(QWidget):

    def __init__(self, qimage=None):
        super(ImagePainter, self).__init__()
        if qimage is None:
            image = QImage(QSize(8, 8), QImage.Format_RGB32)
            image.fill(QColor('black'))
        else:
            image = QImage(qimage)
        self.orig_image = image
        scale_factor_w = (self.size().width() / image.width()) / 8
        scale_factor_h = (self.size().height() / image.height()) / 8
        #scale_factor_w = 2
        #scale_factor_h = 2
        w, h = image.width() * scale_factor_w, image.height() * scale_factor_h
        self.w = w
        self.h = h
        self.set_image(image)
        self.scale_factor_w = scale_factor_w
        self.scale_factor_h = scale_factor_h

        self.pixel_chain = False

    def paintEvent(self, event):

        size = self.size()
        self.x = (size.width() - self.image.size().width()) / 2
        self.y = (size.height() - self.image.size().height()) / 2
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(self.x, self.y, self.image)
        painter.end()

    def mousePressEvent(self, event):
        self.pixel_chain = True
        self.button = event.button()

    def mouseMoveEvent(self, event):
        if self.pixel_chain:
            self._draw(event)

    def mouseReleaseEvent(self, event):
        self.pixel_chain = False

    def _draw(self, event):
        color = QColor('white') if self.button == 1 else QColor('black')
        pos = event.pos()
        x, y = pos.x() - self.x, pos.y() - self.y
        cx, cy = x / self.scale_factor_w, y / self.scale_factor_h
        painter = QPainter()
        painter.begin(self.image)
        painter.setBrush(QBrush(color))
        painter.drawRect(cx * self.scale_factor_w, cy * self.scale_factor_h,
                         self.scale_factor_w, self.scale_factor_h)
        painter.end()
        self.repaint()

        self.window.apply_event(self.layer_name)

    def set_image(self, qimage):
        self.image = qimage.scaled(QSize(self.w, self.h))
        self.repaint()

    def get_image(self):
        return self.image.scaled(self.orig_image.size())

    def sizeHint(self):
        return QSize(800, 800)


def launch(capsule, data, w, h, c, max_feature_maps=np.inf):
    app = QApplication(sys.argv)
    window = Window(capsule, data, w, h, c, max_feature_maps=max_feature_maps)
    window.show()
    sys.exit(app.exec_())
