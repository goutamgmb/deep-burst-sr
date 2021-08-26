import os
import admin.loading as loading
from admin.environment import env_settings


def load_network(net_path, return_dict=False, **kwargs):
    kwargs['backbone_pretrained'] = False
    if os.path.isabs(net_path):
        path_full = net_path
        net, checkpoint_dict = loading.load_network(path_full, **kwargs)
    else:
        path_full = os.path.join(env_settings().workspace_dir, 'checkpoints', net_path)
        net, checkpoint_dict = loading.load_network(path_full, **kwargs)

    if return_dict:
        return net, checkpoint_dict
    else:
        return net
