import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    base_path = os.path.join(os.path.dirname(__file__), '..')
    env_file = os.path.join(os.path.dirname(__file__), 'local.py')
    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_nets_dir': '\'{}/pretrained_networks/\''.format(base_path),
        'save_data_path': empty_str,
        'zurichraw2rgb_dir': empty_str,
        'burstsr_dir': empty_str,
        'synburstval_dir': empty_str
    })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.',
               'pretrained_nets_dir': 'Directory for pre-trained networks.',
               'save_data_path': 'Directory for saving network predictions for evaluation.',
               'zurichraw2rgb_dir': 'Zurich RAW 2 RGB path',
               'burstsr_dir': 'BurstSR dataset path',
               'synburstval_dir': 'SyntheticBurst validation set path'}

    with open(env_file, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')
        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
