import contextlib
import logging
from typing import Dict, Optional, Union

import torch.nn as nn
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

wandb_available = False
with contextlib.suppress(ModuleNotFoundError):
    import wandb

    wandb_available = True

_LOGGER = logging.getLogger(__file__)

def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class SparseMLWrapper(object):
    """
    A wrapper for sparsification of Yolact models with SparseML

    :param model: The model to sparsify
    :param recipe: SparseZoo stub or path to a local training recipe
    :param checkpoint_recipe: A checkpoint recipe if present
    :param checkpoint_epoch: The epoch contained in the checkpoint if any,
        defaults to float('inf')
    """
    def __init__(
        self,
        model: "Module",
        recipe: str,
        checkpoint_recipe: Optional[str] = None,
        checkpoint_epoch: Union[int, float] = float('inf'),
        metadata: Optional[Dict[str, any]] = None,
    ):
        self.enabled = bool(recipe)
        self.model = model.module if is_parallel(model) else model
        self.recipe = recipe
        self.manager = ScheduledModifierManager.from_yaml(
            file_path=recipe,
            metadata=metadata,
        ) if self.enabled else None
        self.logger = None
        self.checkpoint_recipe_manager = ScheduledModifierManager.from_yaml(
            checkpoint_recipe
        ) if checkpoint_recipe else None

        if self.checkpoint_recipe_manager:
            _LOGGER.info("Applying structure from checkpoint recipe")
            self.checkpoint_recipe_manager.apply_structure(
                module=model,
                epoch=checkpoint_epoch,
        )

    def state_dict(self):
        """
        :return: A dict object with composed recipe
        """
        return {
            'recipe': str(self.compose_recipes()) if self.enabled else None,
        }

    def apply(self):
        """
        Apply training recipe to model
        """
        if self.enabled:
            self.manager.apply(self.model)

    def initialize(self, start_epoch: Union[int, float]):
        """
        Initialize manager to a epoch

        :param start_epoch: The epoch to initialize manager at
        """
        if self.enabled:
            self.manager.initialize(self.model, start_epoch)

    def initialize_loggers(self, logger, tb_writer, wandb_logger, rank):
        self.logger = logger
        if not self.enabled or rank not in [-1, 0]:
            return

        def _logging_lambda(tag, value, values, step, wall_time, **kwargs):
            if not wandb_logger or not wandb_available:
                return

            if value is not None:
                wandb.log({tag: value})

            if values:
                wandb.log(values)

        self.manager.initialize_loggers([
            SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                tensorboard=tb_writer,
                wandb_={'project': 'yolact'} if wandb_logger else None,
            )
        ])

        if wandb_logger and wandb_available:
            artifact = wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb.log_artifact(artifact)

    def log_losses_wandb(self, losses=None):
        if wandb_available:
            wandb.log(
                {'losses': losses}
            )

    def modify(self, scaler, optimizer, model, dataloader):
        if self.enabled:
            return self.manager.modify(
                model,
                optimizer,
                steps_per_epoch=len(dataloader),
                wrap_optim=scaler
            )

        return scaler

    def check_lr_override(self, scheduler):
        # Override lr scheduler if recipe makes any LR updates
        if self.should_override_scheduler():
            self.logger.info(
                'Disabling LR scheduler, managing LR using SparseML recipe'
            )
            scheduler = None

        return scheduler

    def should_override_scheduler(self):
        return self.enabled and self.manager.learning_rate_modifiers

    def check_epoch_override(self, epochs) -> Union[int, float]:
        """
        Override num epochs if recipe explicitly modifies epoch range

        :param epochs: potential epoch candidate to override
        :return: The max epoch number to use
        """
        if self.enabled and self.manager.epoch_modifiers and \
                self.manager.max_epochs:
            epochs = self.manager.max_epochs or epochs  # override num_epochs
            self.logger.info(
                f'Overriding number of epochs from SparseML manager to {epochs}'
            )

        return epochs

    def qat_active(self, epoch) -> bool:
        """
        :param epoch: Epoch number to test for
        :return: True if qat active at specified epoch else False
        """
        if self.enabled and self.manager.quantization_modifiers:
            qat_start = min(
                [mod.start_epoch for mod in self.manager.quantization_modifiers]
            )

            return qat_start < epoch + 1

        return False

    def compose_recipes(self):
        """
        Combine checkpoint recipe + training recipe and return the new composed manager

        :return: A manager representing composed recipe
        """
        if not self.checkpoint_recipe_manager:
            return self.manager

        return self.manager.compose_staged(
            base_recipe=self.checkpoint_recipe_manager, additional_recipe=self.recipe,
        )

