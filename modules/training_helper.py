import tensorflow as tf
import numpy as np
from modules.feature_generator import load_dataset
from modules.image_processor import image_augmentation, evenly_rotate
import matplotlib.pyplot as plt
import imageio


def draw_profile_chart(profile, pred_profile, calculated_Vmax, calculated_R34):
    km = np.arange(0, 751, 5)
    plt.figure(figsize=(15, 10), linewidth=2)
    plt.plot(km, profile, color='r', label="profile")
    plt.plot(km, pred_profile, color='g', label="pred_profile")
    plt.axhline(y=calculated_Vmax, color='b', linestyle='-')
    plt.axvline(x=calculated_R34, color='y', linestyle='-')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 120)
    plt.xlabel("Radius", fontsize=30)
    plt.ylabel("Velocity", fontsize=30)
    plt.legend(loc="best", fontsize=20)
    plt.savefig('tmp.png')
    plt.close('all')
    RGB_matrix = imageio.imread('tmp.png')
    return RGB_matrix


def output_sample_profile_chart(profiler, sample_data, summary_writer, epoch_index):
    for phase, (sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34) in sample_data.items():
        pred_profile, calculated_Vmax, calculated_R34 = profiler(sample_images, sample_feature, training=False)
        charts = []
        for i in range(10):
            charts.append(
                draw_profile_chart(sample_profile[i], pred_profile[i], calculated_Vmax[i], calculated_R34[i])
            )
        chart_matrix = np.stack(charts).astype(np.int)
        with summary_writer.as_default():
            tf.summary.image(
                f'{phase}_profile_chart',
                chart_matrix.astype(np.float)/255,
                step=epoch_index,
                max_outputs=chart_matrix.shape[0]
            )


def get_sample_data(dataset, count):
    for batch_index, (images, feature, profile, Vmax, R34) in dataset.enumerate():
        valid_profile = profile[:, 0] != -999
        preprocessed_images = image_augmentation(images)
        sample_images = tf.boolean_mask(preprocessed_images, valid_profile)[:count, ...]
        sample_feature = tf.boolean_mask(feature, valid_profile)[:count, ...]
        sample_profile = tf.boolean_mask(profile, valid_profile)[:count, ...]
        sample_Vmax = tf.boolean_mask(Vmax, valid_profile)[:count, ...]
        sample_R34 = tf.boolean_mask(R34, valid_profile)[:count, ...]
        return sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34


def blend_profiles(profiles):
    return sum(profiles)/len(profiles)


def rotation_blending(model, blending_num, images, feature):
    evenly_rotated_images = evenly_rotate(images, blending_num)
    profiles = []
    Vmax = []
    R34 = []
    for image in evenly_rotated_images:
        pred_profile, pred_Vmax, pred_R34 = model(image, feature, training=False)
        profiles.append(pred_profile)
        Vmax.append(pred_Vmax)
        R34.append(pred_R34)
    return blend_profiles(profiles), tf.reduce_mean(Vmax, 0), tf.reduce_mean(R34, 0)


def evaluate_loss(model, dataset, loss_function, blending_num=10):
    if loss_function == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()

    avg_profile_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    avg_Vmax_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    avg_R34_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for batch_index, (images, feature, profile, Vmax, R34) in dataset.enumerate():
        pred_profile, pred_Vmax, pred_R34 = rotation_blending(model, blending_num, images, feature)
        batch_profile_loss = loss(profile, pred_profile)
        avg_profile_loss.update_state(batch_profile_loss)
        batch_Vmax_loss = loss(Vmax, pred_Vmax)
        avg_Vmax_loss.update_state(batch_Vmax_loss)
        batch_R34_loss = loss(R34, pred_R34)
        avg_R34_loss.update_state(batch_R34_loss)

    profile_loss = avg_profile_loss.result()
    Vmax_loss = avg_Vmax_loss.result()
    R34_loss = avg_R34_loss.result()

    return profile_loss, Vmax_loss, R34_loss


def get_tensorflow_datasets(data_folder, batch_size, shuffle_buffer, good_VIS_only=False, valid_profile_only=False, coordinate='cart'):
    datasets = dict()
    for phase in ['train', 'valid', 'test']:
        phase_data = load_dataset(data_folder, phase, good_VIS_only, valid_profile_only, coordinate)
        images = tf.data.Dataset.from_tensor_slices(phase_data['image'])

        feature = tf.data.Dataset.from_tensor_slices(
            phase_data['feature'].to_numpy(dtype='float32')
        )

        profile = tf.data.Dataset.from_tensor_slices(phase_data['profile'])

        Vmax = tf.data.Dataset.from_tensor_slices(
            phase_data['label'][['Vmax']].to_numpy(dtype='float32')
        )

        R34 = tf.data.Dataset.from_tensor_slices(
            phase_data['label'][['R34']].to_numpy(dtype='float32')
        )

        datasets[phase] = tf.data.Dataset.zip((images, feature, profile, Vmax, R34)) \
            .shuffle(shuffle_buffer) \
            .batch(batch_size)

    return datasets
