* Dataset:
    - Translation range
    - Rotation range
    - Image size
    - Pre-processing:
        - Scaling pixels to 0..1
        - Scaling pixels to -1..1
* Architecture:
    - DeepConvnet:
        - nb of layers
        - for each layer :
            - size of filters
            - nb of filters
            - non linearity (sigmoid, tanh, rectify, thresh_linear,  leaky_rectify, very_leaky_rectify, linear)
            - padding
            - activate bias or not
        - Tiying encoder/decoder layers or not
        - Winner take all layer (on each region take the max over channels, put zero in other channels) after conv layers 
        - A Sum layer at the end to "generate"
        - pad='same' everywhere (to keep dimensions of all feature maps equal to input)
        - Convolutions with stride=2 (like pooling) on padded inputs ("slow" pooling)

* Optimization:
    - Algorithm (rmsprop, adadelta, adam, adagrad, momentum, nesterov_momentum)
    - Global learning rate
    - Global learning rate schedule:
        * Exponential decay : **decay parameter**
        * Linear decay : **decay parameter**
        * Manual decay (epoch **N1**: divide global learning rate by **X**, epoch **N2**: divide global learning rate by **X**...)
        * Slow decay (if validation/train do not improve on **patience** epochs divide global learning rate by **X**)
        * **Momentum** if supported by algorithm
    - nb of epochs of patience
    - max nb of epochs
    - min nb of epochs
* Regularization:
    - Regularization on filters :
        - **L1** : coeficient
        - **L2** : coeficient
        - Norm constraint (**max norm value**)

* Initialization:
    - Weight initalization method (Orthogonal, Glorot, He, Manual, Sparse) and Distribution (Normal, Gaussian)

* Loss:
	- Reconstruction error:
		- squared error
		- cross entropy
