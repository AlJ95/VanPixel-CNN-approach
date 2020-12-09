import random as r
import cv2
import numpy as np
from skimage.util import random_noise


def create_figs(image, image_pre_result, no_of_objects=1):
    for i in range(no_of_objects):
        width = r.randint(0, image.shape[0] / 2)
        height = r.randint(0, image.shape[1] / 2)
        xpos = r.randint(0, image.shape[1])
        ypos = r.randint(0, image.shape[0])
        group_number = i + 1
        rgb = np.array([r.random(), r.random(), r.random(), 1])

        randInt = r.randint(0, 1)
        if randInt == 0:
            cv2.ellipse(image, (xpos, ypos), (width, height), 0,
                        0, 360, rgb, -1)
            cv2.ellipse(image_pre_result, (xpos, ypos),
                        (width, height), 0, 0, 360,
                        (group_number + 1) * 25, -1)
        else:
            cv2.rectangle(image, (xpos, ypos),
                          (xpos + width, ypos + height), rgb, -1)
            cv2.rectangle(image_pre_result, (xpos, ypos),
                          (xpos + width, ypos + height),
                          (group_number + 1) * 25, -1)

    return image, image_pre_result


def produce_image(no_of_images, sp_prob=0.05, dim=(1080, 1920, 4),
                  grad="Laplacian", kernel_size=3):

    image = np.ones(dim)
    image_pre_result = np.full(image.shape[0:2], 1)

    image, image_pre_result = create_figs(image, image_pre_result,
                                          no_of_images)

    if grad == "Laplacian":
        gradient = cv2.Laplacian(image.astype('float64'), cv2.CV_64F,
                                 ksize=kernel_size)[:, :, 0:3].clip(0, 1)
    elif grad == "Scharr":
        dx = cv2.Scharr(image.astype('float64'),
                        cv2.CV_64F, dx=1, dy=0)[:, :, 0:3]
        dy = cv2.Scharr(image.astype('float64'),
                        cv2.CV_64F, dx=0, dy=1)[:, :, 0:3]
        gradient = np.where(np.abs(dx) >= np.abs(dy),
                            np.abs(dx), np.abs(dy)).clip(0, 1)
    else:
        dx = cv2.Sobel(image.astype('float64'), cv2.CV_64F,
                       dx=1, dy=0, ksize=kernel_size)[:, :, 0:3]
        dy = cv2.Sobel(image.astype('float64'), cv2.CV_64F,
                       dx=0, dy=1, ksize=kernel_size)[:, :, 0:3]
        gradient = np.where(np.abs(dx) >= np.abs(dy), np.abs(dx),
                            np.abs(dy)).clip(0, 1)

    gradientRGB = np.amax(gradient, axis=2)

    if sp_prob > r.random():
        image = random_noise(image, mode='s&p', amount=0.05)
        gradient = cv2.Laplacian(image.astype('float64'),
                                 cv2.CV_64F, ksize=11)[:, :, 0:3]
        gradient = np.abs(gradient)
        gradient /= np.max(gradient)

    gradientRGB = np.where(abs(gradientRGB) < 0.1, 0, 1)
    return image, gradient, gradientRGB
