import tensorflow as tf
from collections import defaultdict
from modules.training_helper import evaluate_loss, get_sample_data, output_sample_profile_chart
from modules.image_processor import image_augmentation


def calculate_pred_V_at_R34(R34, pred_profile):
    R34_index = tf.cast(tf.round(R34/5), tf.int32)
    valid_R34 = tf.math.logical_and(0 < R34_index, R34_index <= 150)
    pred_V_at_R34 = tf.expand_dims(
        tf.gather_nd(
            pred_profile,
            tf.clip_by_value(R34_index, 0, 150),
            batch_dims=1
        ),
        axis=1
    )
    pred_V_at_R34 = tf.boolean_mask(pred_V_at_R34, valid_R34)
    return pred_V_at_R34


def train_profiler(
    profiler,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    early_stop_tolerance=None,
    overfit_tolerance=None,
    loss_function='MSE',
    loss_ratio={}
):
    optimizer = tf.keras.optimizers.Adam()
    if loss_function == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(image, feature, profile, Vmax, R34):
        with tf.GradientTape() as tape:
            pred_profile, calculated_Vmax, calculated_R34 = profiler(image, feature, training=True)

            have_valid_profile = tf.cast(profile[:, 0] != -1, tf.float32)
            profile_loss = loss(profile, pred_profile, sample_weight=have_valid_profile)
            Vmax_loss = loss(Vmax, calculated_Vmax)

            R34_loss = loss(R34, calculated_R34)

            profiler_loss = loss_ratio.get('profile', 0.0) * profile_loss \
                + loss_ratio.get('Vmax', 0.0) * Vmax_loss \
                + loss_ratio.get('R34', 0.0) * R34_loss

            if loss_ratio.get('V_at_R34', 0.0):
                pred_V_at_R34 = calculate_pred_V_at_R34(R34, pred_profile)
                V_at_R34_loss = loss(34, pred_V_at_R34)
                profiler_loss += loss_ratio.get('V_at_R34', 0.0) * V_at_R34_loss
            else:
                V_at_R34_loss = 0

        gradients = tape.gradient(profiler_loss, profiler.trainable_variables)
        # gradients = [tf.clip_by_norm(g, 2) for g in gradients]

        optimizer.apply_gradients(zip(gradients, profiler.trainable_variables))

        avg_losses[f'profiler: profile_{loss_function}_loss'].update_state(profile_loss)
        avg_losses[f'profiler: Vmax_{loss_function}_loss'].update_state(Vmax_loss)
        avg_losses[f'profiler: R34_{loss_function}_loss'].update_state(R34_loss)
        avg_losses[f'profiler: V_at_R34_{loss_function}_loss'].update_state(V_at_R34_loss)
        avg_losses[f'profiler: overall_{loss_function}_loss'].update_state(profiler_loss)
        return

    sample_data = {
        phase: get_sample_data(datasets[phase], 10)
        for phase in ['train', 'valid']
    }

    # use stack to keep track on validation loss and help early stopping
    valid_loss_stack = []
    for epoch_index in range(1, max_epoch+1):
        print(f'Executing epoch #{epoch_index}')
        for batch_index, (images, feature, profile, Vmax, R34) in datasets['train'].enumerate():
            preprocessed_images = image_augmentation(images)
            train_step(preprocessed_images, feature, profile, Vmax, R34)

        with summary_writer.as_default():
            for loss_name, avg_loss in avg_losses.items():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
            avg_loss.reset_states()

            for variable in profiler.variables:
                if variable.name in ['calibrate_factor:0', 'calibrate_constant:0']:
                    tf.summary.scalar(variable.name, variable.numpy(), step=epoch_index)

        if (epoch_index) % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')
            # draw profile chart
            if epoch_index % 20 == 0:
                output_sample_profile_chart(profiler, sample_data, summary_writer, epoch_index)
            # calculate blending loss
            indicator_for_early_stop = {}
            for phase in ['train', 'valid']:
                profile_loss, Vmax_loss, R34_loss = evaluate_loss(profiler, datasets[phase], loss_function)
                with summary_writer.as_default():
                    for name, loss in [('profile', profile_loss), ('Vmax', Vmax_loss), ('R34', R34_loss)]:
                        tf.summary.scalar(f'[{phase}] profiler: blending_{name}_loss', loss, step=epoch_index)
                indicator_for_early_stop[phase] = profile_loss
            # save the best profiler and check for early stopping
            while valid_loss_stack and valid_loss_stack[-1] >= indicator_for_early_stop['valid']:
                valid_loss_stack.pop()
            if not valid_loss_stack:
                profiler.save_weights(saving_path + '/profiler', save_format='tf')
                print('Get the best validation performance so far! Saving the model.')
            elif early_stop_tolerance and len(valid_loss_stack) > early_stop_tolerance:
                print('Exceed the early stop tolerance, training procedure will end!')
                break
            elif overfit_tolerance and (indicator_for_early_stop['valid'] - indicator_for_early_stop['train']) >= overfit_tolerance:
                print('Exceed the orverfit tolerance, training procedure will end!')
                # since valid loss is using blending, if train loss can beat valid loss,
                # that probably means profiler is already overfitting.
                break
            valid_loss_stack.append(indicator_for_early_stop['valid'])
