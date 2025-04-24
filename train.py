import os
current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)
# os.environ['HF_HUB_CACHE'] = '.cache/huggingface/hub'

from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=False)
        print(f'resume from {config.model.resume}')

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.val_check_interval = float(10.0)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
