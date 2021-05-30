# Inverse Lukas-Kanade
"Efficient" implementation of the inverse Lucas-Kanade with a translation-only window. This work is based on the inverse compositional Lucas Kanade from Baker, Simon & Matthews, Iain. (2004). Lucas-Kanade 20 Years On: A Unifying Framework Part 1: The Quantity Approximated, the Warp Update Rule, and the Gradient Descent Approximation. International Journal of Computer Vision - IJCV.
Differently than the original is that this repository only considers a translation window only. Therefore, the calculations are slightly simplified and requires less computation. Furthermore, it is programmed in C++ whereas the authors provide their code in MATLAB.
This code works only with the correct inputs. Have a look at LK_class_example.cpp. For example, the image pyramid must be created in a similar way as the example, because some parts are hard coded.
This algorithm is used/tested in a modified Visual Inertial Odometry algorithm (Stereo MSCKF-VIO) https://github.com/sbahnam/msckf_vio. The main benefit is that it requires less computation time, especially when tracking a small number of features. Furthermore, it showed a similar tracking performance on the EuRoC dataset. In the VIO the algorithm the inverse LK is implemented as a function. However, using the class implementation has some computational benefits, while still perform similarly.
