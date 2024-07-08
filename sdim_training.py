import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from dl_project.dataset.shapes3D import Shapes3D
from dl_project.losses.SDIM_loss import SDIMLoss
from dl_project.trainer.SDIM_trainer import SDIM, SDIMTrainer
from ruamel.yaml import YAML

###### Shared Representation Learning

conf_path = './conf/share_conf.yaml'
with open(conf_path, "r") as f:
    yaml = YAML(typ='safe', pure=True)
    conf = yaml.load(f)
    
TRAINING_PARAM = conf["training_param"]
MODEL_PARAM = conf["model_param"]
LOSS_PARAM = conf["loss_param"]

sdim = SDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=MODEL_PARAM["shared_dim"],
    )
loss = SDIMLoss(
    local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
    global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
    shared_loss_coeff=LOSS_PARAM["shared_loss_coeff"],
)

train_dataset = Shapes3D()

device = TRAINING_PARAM["device"]
learning_rate = TRAINING_PARAM["learning_rate"]
batch_size = TRAINING_PARAM["batch_size"]
epochs = TRAINING_PARAM["epochs"]
trainer = SDIMTrainer(
    dataset_train=train_dataset,
    model=sdim,
    loss=loss,
    learning_rate=learning_rate,
    batch_size=batch_size,
    device=device,
)
timestamp = time.time()
trainer.train(epochs=epochs, xp_name=f'SDIM')
