import glob
import os
import h5py
import re

import yaml
from PyInquirer import prompt
from prompt_toolkit.validation import Validator, ValidationError

CONFIG_PATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'config_train.yml')


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a integer number',
                cursor_position=len(document.text))  # Move cursor to end


class FloatValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a float number',
                cursor_position=len(document.text))  # Move cursor to end


def string_to_int_list(_string):
    return [int(_int) for _int in re.findall(r'\b\d+\b', _string)]


class IntListValidator(Validator):
    def validate(self, document):
        try:
            assert len(string_to_int_list(document.text)) == 3
        except:
            raise ValidationError(
                message='Please enter a list in the format (1, 64, 64) or 1, 64, 64 or [1, 64, 64]',
                cursor_position=len(document.text))  # Move cursor to end


def _filter_stacks(path):
    with h5py.File(path, 'r') as f:
        _keys = list(f.keys())

        if 'raw' not in _keys:
            print(f" -file {path}, will be ignored because it does not contain a <raw> dataset")
            return False

        if 'label' not in _keys:
            print(f" -file {path}, will be ignored because it does not contain a <label> dataset")
            return False

        if f['raw'].shape != f['label'].shape:
            print(f" -file {path}, will be ignored because "
                  f"it has raw and label have different shapes ({f['raw'].shape}, {f['label'].shape})")
            return False

    return True


def get_key(_dict, _list_key):
    if len(_list_key) == 1:
        return _dict[_list_key[0]]
    else:
        return get_key(_dict[_list_key[0]], _list_key[1:])


def assign_key(value, _dict, _list_key):
    if len(_list_key) == 1:
        _dict[_list_key[0]] = value
        return _dict
    else:
        _dict[_list_key[0]] = assign_key(value, _dict[_list_key[0]], _list_key[1:])
        return _dict


_question_save_path = {'type': 'input',
                       'name': 'save_path',
                       'message': "Where do you want to save the config",
                       'default': './new_experiment'}

_question_model_name = {'type': 'list',
                        'name': None,
                        'message': 'Do you want to train a 2D or 3D model?\n',
                        'choices': ['UNet3D', 'UNet2D']}

_question_f_maps = {'type': 'input',
                    'name': None,
                    'message': 'Change the number of initial features?\n'
                               '(Higher number means more powerful network at the cost of speed and memory)',

                    'validate': NumberValidator,
                    'filter': lambda val: int(val)
                    }

_question_learning_rate = {'type': 'input',
                           'name': None,
                           'message': 'Change the learning rate?\n (Smaller slower convergence)',
                           'validate': FloatValidator,
                           'filter': lambda val: float(val)
                           }

_question_weight_decay = {'type': 'input',
                          'name': None,
                          'message': 'Change the weight decay?\n (Larger weight decay reduce over-fitting)',
                          'validate': FloatValidator,
                          'filter': lambda val: float(val)
                          }

_question_checkpoint_dir = {'type': 'input',
                            'name': None,
                            'message': 'Change the checkpoint dir?'}

_question_files_train = {'type': 'checkbox',
                         'name': None,
                         'message': 'Select the files to use for training:',
                         'choices': None,
                         'filter': lambda var: list(filter(_filter_stacks, var))}

_question_files_val = {'type': 'checkbox',
                       'name': None,
                       'message': "Select the files to use for validation:\n"
                                  " (training files can not be used for validation)",
                       'choices': None,
                       'filter': lambda var: list(filter(_filter_stacks, var))}

_question_patch_train = {'type': 'input',
                         'name': None,
                         'message': 'Change the default training patch size?\n'
                                    ' (If a 2D model has been selected <z> must have path size 1, e.g [1, 256, 256])',
                         'validate': IntListValidator,
                         'filter': lambda var: string_to_int_list(var)}

_question_patch_val = {'type': 'input',
                       'name': None,
                       'message': 'Change the default validation patch size?\n'
                                  ' (If a 2D model has been selected <z> must have path size 1, e.g [1, 256, 256])',
                       'validate': IntListValidator,
                       'filter': lambda var: string_to_int_list(var)}

_question_stride_train = {'type': 'input',
                          'name': None,
                          'message': 'Change the default training stride size?\n'
                                     ' - (Stride size should be smaller than patch size in order to cover the all '
                                     'volume)\n'
                                     ' - (If a 2D model has been selected <z> must have stride size 1, '
                                     'e.g [1, 256, 256])\n',
                          'validate': IntListValidator,
                          'filter': lambda var: string_to_int_list(var)}

_question_stride_val = {'type': 'input',
                        'name': None,
                        'message': 'Change the default validation stride size?\n'
                                   ' - (Stride size should be smaller than patch size in order to cover the all '
                                   'volume)\n'
                                   ' - (If a 2D model has been selected <z> must have stride size 1, '
                                   'e.g [1, 256, 256])\n',
                        'validate': IntListValidator,
                        'filter': lambda var: string_to_int_list(var)}

_question_batch_size = {'type': 'input',
                        'name': None,
                        'message': 'Change the default batch size?\n'
                                   ' - (Bigger batch size improve learning stability and (sometimes) performance,'
                                   ' but requires a lot of GPU memory.) \n',
                        'validate': IntListValidator,
                        'filter': lambda var: string_to_int_list(var)}

def train_configurator_wizard(data_path, config=None):

    if config is None:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    mappings = {'model_name': (['model', 'name'], _question_model_name),
                'f_maps': (['model', 'f_maps'], _question_f_maps),
                'learning_rate': (['optimizer', 'learning_rate'], _question_learning_rate),
                'weight_decay': (['optimizer', 'weight_decay'], _question_weight_decay),
                'checkpoint_dir': (['trainer', 'checkpoint_dir'], _question_checkpoint_dir),
                'batch_size': (['loaders', 'batch_size'], _question_batch_size),
                'files_train': (['loaders', 'train', 'file_paths'], _question_files_train),
                'patch_train': (['loaders', 'train', 'slice_builder', 'patch_shape'], _question_patch_train),
                'stride_train': (['loaders', 'train', 'slice_builder', 'stride_shape'], _question_stride_train),
                'files_val': (['loaders', 'val', 'file_paths'], _question_files_val),
                'patch_val': (['loaders', 'val', 'slice_builder', 'patch_shape'], _question_patch_val),
                'stride_val': (['loaders', 'val', 'slice_builder', 'stride_shape'], _question_stride_val),
                }

    possible_paths = glob.glob(f"{data_path}/**/*.h5", recursive=True)
    possible_paths = [{'name': path} for path in possible_paths]
    all_questions = [_question_save_path]
    for name, (h_keys, question) in mappings.items():
        question['name'] = name
        if name == 'files_train' or name == 'files_val':
            question['choices'] = possible_paths
        else:
            question['default'] = str(get_key(config, h_keys))

        all_questions.append(question)

    answers = prompt(all_questions)

    save_path = answers['save_path']
    del answers['save_path']

    for name, value in answers.items():
        h_keys, _ = mappings[name]
        config = assign_key(value, config, h_keys)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config_train.yml", "w") as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    train_configurator_wizard("/home/lcerrone/datasets/ovules")
