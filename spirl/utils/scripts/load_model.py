from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.utils.general_utils import AttrDict
from spirl.components.checkpointer import get_config_path
from spirl.utils.general_utils import map_dict
from spirl.rl.components.agent import BaseAgent
from spirl.utils.pytorch_utils import no_batchnorm_update
import os
import torch
import imp
import numpy as np

def load_model(mode, batch_size):
    assert mode in ['cl', 'ol']

    if mode == 'ol':
        model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/kitchen/hierarchical_tmp")
    elif mode == 'cl':
        model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/kitchen/hierarchical_cl_tmp")

    model_params = AttrDict(
        state_dim=data_spec.state_dim,
        action_dim=data_spec.n_actions,
        kl_div_weight=5e-4,
        nz_enc=128,
        nz_mid=128,
        n_processing_layers=5,
        nz_vae=10,
        n_rollout_steps=10,
    )

    if mode == 'cl':
        model_params.cond_decode = True

    device = torch.device('cuda')

    model_params['batch_size'] = batch_size
    model_params.update(data_spec)
    model_params['device'] = device

    if mode == 'ol':
        model = SkillPriorMdl(model_params, logger=None)
    elif mode == 'cl':
        model = ClSPiRLMdl(model_params, logger=None)
    model = model.to(device)
    BaseAgent.load_model_weights(model, model_checkpoint)

    return model


def eval_model():
    # mode = 'ol'
    mode = 'cl'

    batch_size = 128

    model = load_model(mode, batch_size)

    # get some data to try things out on
    dataset_class = data_spec.dataset_class
    conf_path = get_config_path('../../configs/skill_prior_learning/kitchen/hierarchical')
    data_config = imp.load_source('conf', conf_path).data_config
    data_config['device'] = model.device.type

    data_loader = dataset_class(data_dir='.', data_conf=data_config, resolution=64,
                                phase="train", shuffle=True, dataset_size=-1). \
        get_data_loader(batch_size=batch_size, n_repeat=10)

    for batch_idx, sampled_batch in enumerate(data_loader):
        print(batch_idx)
        states = sampled_batch['states']
        actions = sampled_batch['actions']

        inputs = AttrDict(map_dict(lambda x: x.to(model.device), sampled_batch))

        with no_batchnorm_update(model):
            model_outputs = model(inputs)
            losses = model.loss(model_outputs, inputs)
            sq_diff = np.square((model_outputs.reconstruction - inputs.actions).cpu().detach().numpy())
            print(np.mean(sq_diff))

if __name__ == "__main__":
    eval_model()