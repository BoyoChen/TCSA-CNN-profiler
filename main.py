import argparse
import os
import importlib
import tensorflow as tf

from modules.profiler_trainer import train_profiler
from modules.training_helper import evaluate_loss, get_tensorflow_datasets
from modules.experiment_helper import parse_experiment_settings


def prepare_model_save_path(experiment_name, sub_exp_name):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')

    saving_folder = 'saved_models/' + experiment_name
    if not os.path.isdir(saving_folder):
        os.mkdir(saving_folder)

    model_save_path = saving_folder + '/' + sub_exp_name
    return model_save_path


def create_model_by_experiment_settings(experiment_settings, load_from=''):

    def create_model_instance(model_category, model_name):
        model_class = importlib.import_module(f'model_library.{model_category}s.{model_name}').Model
        return model_class()

    profiler = create_model_instance('profiler', experiment_settings['profiler'])
    if load_from:
        profiler.load_weights(f'{load_from}/profiler')
    return profiler


# This function is faciliating creating model instance in jupiter notebook
def create_model_by_experiment_path_and_stage(experiment_path, sub_exp_name):
    sub_exp_settings = parse_experiment_settings(experiment_path, only_this_sub_exp=sub_exp_name)
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']

    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)
    model = create_model_by_experiment_settings(sub_exp_settings, load_from=model_save_path)
    return model


def execute_sub_exp(sub_exp_settings, action, run_anyway):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']
    log_path = f'logs/{experiment_name}/{sub_exp_name}'

    print(f'Executing sub-experiment: {sub_exp_name}')
    if not run_anyway and action == 'train' and os.path.isdir(log_path):
        print('Sub-experiment already done before, skipped ಠ_ಠ')
        return

    summary_writer = tf.summary.create_file_writer(log_path)
    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)
    datasets = get_tensorflow_datasets(**sub_exp_settings['data'])

    if action == 'train':
        model = create_model_by_experiment_settings(sub_exp_settings)

        train_profiler(
            model,
            datasets,
            summary_writer,
            model_save_path,
            **sub_exp_settings['train_profiler']
        )

    elif action == 'evaluate':
        model = create_model_by_experiment_settings(sub_exp_settings, load_from=model_save_path)
        for phase in datasets:
            profile_loss, Vmax_loss, R34_loss = evaluate_loss(model, datasets[phase], loss_function='MSE')
            print(f'{phase} MSE profile_loss: {profile_loss}')
            print(f'{phase} MSE Vmax_loss: {Vmax_loss}')
            print(f'{phase} MSE R34_loss: {R34_loss}')


def main(action, experiment_path, GPU_limit, run_anyway):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')

    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )

    # parse yaml to get experiment settings
    experiment_list = parse_experiment_settings(experiment_path)

    for sub_exp_settings in experiment_list:
        execute_sub_exp(sub_exp_settings, action, run_anyway)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', help='name of the experiment setting, should match one of them file name in experiments folder')
    parser.add_argument('--action', help='(train/evaluate)', default='train')
    parser.add_argument('--GPU_limit', type=int, default=3000)
    parser.add_argument('--omit_completed_sub_exp', action='store_true')
    args = parser.parse_args()
    main(args.action, args.experiment_path, args.GPU_limit, (not args.omit_completed_sub_exp))
