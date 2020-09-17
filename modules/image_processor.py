import numpy as np
import tensorflow_addons as tfa
import polarTransform as pt


# work around!!
def _is_polar_coordinate(images):
    batch_size, height, width, channel_num = images.shape
    return (height != width)


def crop_center(image_matrix, crop_width):
    total_width = image_matrix.shape[1]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return image_matrix[:, start:end, start:end, :]


def shift_polar_images(images, shift_distance):
    # batch_size, height, width, channel_num = images.shape

    shifted_images = np.roll(images, shift=int(round(shift_distance)), axis=1)
    return shifted_images

    # shifted_images_list = []
    # for image, shift_distance in zip(images, shift_distances):
    #     shifted_image = np.concatenate([image[round(shift_distance):], image[:round(shift_distance)]])
    #     shifted_images_list.append(shifted_image)

    # return np.stack(shifted_images_list)


def random_rotate(images):
    if _is_polar_coordinate(images):
        shift_distances = np.random.uniform(
            0, images.shape[1]
        )
        rotated_images = shift_polar_images(
            images,
            shift_distances
        )
    else:
        rotate_angle = np.random.uniform(
            0, 360,
            images.shape[0]
        )
        rotated_images = tfa.image.rotate(
            images,
            angles=rotate_angle,
            interpolation='BILINEAR'
        )

    return rotated_images


def image_augmentation(images, crop_width=64):
    rotated_images = random_rotate(images)
    if _is_polar_coordinate(images):
        return rotated_images

    # cart_coordinate image need center cropping
    cropped_images = crop_center(rotated_images, crop_width)
    return cropped_images


def evenly_rotate(images, rotate_num):
    evenly_rotated_images = []
    if _is_polar_coordinate(images):
        theta = float(images.shape[1])
        for shift_distance in np.arange(0, theta, theta/rotate_num):
            rotated_images = shift_polar_images(
                images,
                shift_distance
            )
            evenly_rotated_images.append(rotated_images)
    else:
        for angle in np.arange(0, 360, 360.0/rotate_num):
            rotated_images = tfa.image.rotate(
                images,
                angles=angle,
                interpolation='BILINEAR'
            )
            input_images = crop_center(rotated_images, 64)
            evenly_rotated_images.append(input_images)
    return evenly_rotated_images


def cart2polar(image):
    # 128x128x? -> 180x103x?
    if image.shape[0] != 128 or image.shape[1] != 128:
        print('ERROR! wrong image size: ' + str(image.shape))
        return None
    polarImage, _ = pt.convertToPolarImage(
        image,
        hasColor=True,
        finalRadius=64,
        radiusSize=103,
        angleSize=180
    )
    return polarImage


def polar2cart(image):
    # 180x103x? -> 128x128x?
    if image.shape[0] != 180 or image.shape[1] != 103:
        print('ERROR! wrong image size: ' + str(image.shape))
        return None
    cartesianImage, _ = pt.convertToCartesianImage(
        image,
        hasColor=True,
        finalRadius=64,
        imageSize=(128, 128)
    )
    return cartesianImage
