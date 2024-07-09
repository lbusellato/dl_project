import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from dl_project.dataset.shapes3D import Shapes3D
from dl_project.losses.EDIM_loss import EDIMLoss
from dl_project.neural_networks.encoder import BaseEncoder
from dl_project.trainer.EDIM_trainer import EDIM, EDIMTrainer, freeze_grad_and_eval
from os.path import join
from ruamel.yaml import YAML

###### Exclusive Representation Learning

conf_path = './conf/exclusive_conf.yaml'
with open(conf_path, "r") as f:
    yaml = YAML(typ='safe', pure=True)
    conf = yaml.load(f)

    
TRAINING_PARAM = conf["training_param"]
MODEL_PARAM = conf["model_param"]
LOSS_PARAM = conf["loss_param"]
SHARED_PARAM = conf["shared_param"]

trained_enc_x = BaseEncoder(
    img_size=SHARED_PARAM["img_size"],
    in_channels=MODEL_PARAM["channels"],
    num_filters=64,
    kernel_size=4,
    repr_dim=SHARED_PARAM["shared_dim"],
)
trained_enc_y = BaseEncoder(
    img_size=SHARED_PARAM["img_size"],
    in_channels=MODEL_PARAM["channels"],
    num_filters=64,
    kernel_size=4,
    repr_dim=SHARED_PARAM["shared_dim"],
)

# Load the trained shared encoders
trained_encoder_path = './mlruns/sdim/run1'
trained_enc_x_path=join(trained_encoder_path, "artifacts/sh_encoder_x/state_dict.pth")
trained_enc_y_path=join(trained_encoder_path, "artifacts/sh_encoder_y/state_dict.pth")
trained_enc_x.load_state_dict(torch.load(trained_enc_x_path))
trained_enc_y.load_state_dict(torch.load(trained_enc_y_path))
freeze_grad_and_eval(trained_enc_x)
freeze_grad_and_eval(trained_enc_y)

train_dataset = Shapes3D()

edim = EDIM(
    img_size=MODEL_PARAM["img_size"],
    channels=MODEL_PARAM["channels"],
    shared_dim=SHARED_PARAM["shared_dim"],
    exclusive_dim=MODEL_PARAM["exclusive_dim"],
    trained_encoder_x=trained_enc_x,
    trained_encoder_y=trained_enc_y,
)
loss = EDIMLoss(
    local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
    global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
    disentangling_loss_coeff=LOSS_PARAM["disentangling_loss_coeff"],
)

device = TRAINING_PARAM["device"]
learning_rate = TRAINING_PARAM["learning_rate"]
batch_size = TRAINING_PARAM["batch_size"]
epochs = TRAINING_PARAM["epochs"]
trainer = EDIMTrainer(
    dataset_train=train_dataset,
    model=edim,
    loss=loss,
    learning_rate=learning_rate,
    batch_size=batch_size,
    device=device,
)
trainer.train(epochs=epochs, xp_name=f'EDIM')