import copy
import numpy as np
import numpy

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros(
                (out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros(
                (out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = numpy.zeros(out_shape,
                                                 dtype='uint8' if output_pixel_vals else out_array.dtype
                                                 ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = numpy.zeros(
            out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[
                            tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array


def build_visualizations_by_averaging_inputs_maximizing_activations(inputs, activations,
                                                                    shape_2d=None, top=1.):
    """
    inputs must be (ninputs, dim inputs)
    activations must be (ninputs, nhidden units)
    averaging the inputs with the top 100%*top activations
    """
    if shape_2d is None:
        shape = int(np.sqrt(inputs.shape[1]))
        shape_2d = (shape, shape)

    visualizations = np.zeros( (activations.shape[1], shape_2d[0], shape_2d[1]  ) )

    for i in xrange(activations.shape[1]):
        acts = activations[:, i]
        inds = range(inputs.shape[0])
        inds = sorted(inds, key=lambda i: acts[i], reverse=True)
        last = max(1, int(top * inputs.shape[0]))
        inds = inds[0:last]
        visualizations[i] = inputs[inds].mean(axis=0).reshape((shape_2d[0], shape_2d[1]))
    return visualizations


def build_visualizations_by_weighted_combinations(layers_weights, top=1., shape_2d=None,
                                                  until_layer=0):

    """top is the proportion of the units of the previous layer to take into account
       when building the visualization of the units of the current layer
       if top = 1., take every unit in the previous layer

       if shape_2d is None then assume it is a square and infer it from the weight matrix of the
        first layer"""
    if shape_2d is None:
        shape = int(np.sqrt(layers_weights[0].shape[1]))
        shape_2d = (shape, shape)
    visualizations_cur_layer = layers_weights[0].reshape((layers_weights[0].shape[0],
                                                          shape_2d[0], shape_2d[1]))
    visualizations = [visualizations_cur_layer]
    for layer in xrange(1, until_layer + 1):
        W = layers_weights[layer]
        visualizations_cur_layer = build_visualizations_combinations_from_previous_layer(visualizations_cur_layer, W, top=top)
        visualizations.append(visualizations_cur_layer)

    return visualizations


def build_visualizations_combinations_from_previous_layer(visualizations_prev_layer, W, top=1.):
    """
    visualizations: axis 0 = units, axis=1 w axis=2 h
     W : axis=0 units , axis=1 prev layer units
    """
    visualizations_cur_layer = np.zeros((W.shape[0], visualizations_prev_layer.shape[1],
                                         visualizations_prev_layer.shape[2]))

    for unit in xrange(W.shape[0]):
        weights = copy.copy(W[unit])
        weights_ind = range(W.shape[1])
        weights_ind = sorted(weights_ind, key=lambda i: weights[i], reverse=True)
        lasts = max(int(top*W.shape[1]), 1)
        bottom_weights_ind = weights_ind[lasts:]
        weights[bottom_weights_ind] = 0.
        weights /= np.sum(weights)
        visualizations_cur_layer[unit] = np.sum(visualizations_prev_layer *
                                                weights[:, np.newaxis, np.newaxis], axis=0)
    return visualizations_cur_layer


def grid_plot(visualizations, nbrows=None, nbcols=None, random=False, imshow_options=None, fig=None):
    """
    if nbrows or nbcols are None, they are inferred from the number of visualizations
    (by taking the sqr root)
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    if imshow_options is None:
        imshow_options = {}
    if nbrows is None:
        nbrows = int(np.sqrt(visualizations.shape[0])) + 1
    if nbcols is None:
        nbcols = int(np.sqrt(visualizations.shape[0])) + 1

    for row in xrange(nbrows):
        for col in xrange(nbcols):

            k = (col + row * nbcols)
            if random is True:
                ind = np.random.randint(0, visualizations.shape[0] - 1)
            else:
                ind = k
            if ind >= visualizations.shape[0]:
                break
            ax = fig.add_subplot(nbrows, nbcols, k + 1)
            ax.axis('off')
            ax.imshow(visualizations[ind], **imshow_options)

def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape = None, normalize=True):
    """ Display an array of rgb images. 
    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = numpy.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()

    if normalize:
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    if shape is None:
        n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]
        
    im = numpy.array(bordercolor)*numpy.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = numpy.concatenate((
                  numpy.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*numpy.ones((height,border,3),dtype=float)), 1),
                  bordercolor*numpy.ones((border,width+border,3),dtype=float)
                  ), 0)
    return im

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    M = np.random.uniform(size=(100, 5, 5, 3)) * 100
    img = dispims_color(M, border=1)
    plt.imshow(img, interpolation='none')
    plt.show()
