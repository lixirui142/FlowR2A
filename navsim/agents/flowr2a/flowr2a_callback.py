import torch
import pytorch_lightning as pl

from navsim.agents.flowr2a.flowr2a_config import TransfuserConfig


class TransfuserCallback(pl.Callback):
    """Training callback: logs the learning rate and (once) flags parameters with no gradient."""

    def __init__(
        self,
        config: TransfuserConfig,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        self._config = config
        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Log the learning rate."""
        lr = next(iter(optimizer.param_groups))['lr']
        lightning_module.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)

        # # Debug: detect parameters unused in loss computation (causes DDP errors)
        # if not hasattr(self, '_unused_params_checked'):
        #     self._unused_params_checked = True
        #     unused = [
        #         name for name, param in lightning_module.named_parameters()
        #         if param.requires_grad and param.grad is None
        #     ]
        #     if unused:
        #         print(f"\n{'='*60}")
        #         print(f"[DDP DEBUG] {len(unused)} parameters have requires_grad=True but got no gradient:")
        #         for name in unused:
        #             print(f"  - {name}")
        #         print(f"{'='*60}\n")
        #     else:
        #         print("[DDP DEBUG] All requires_grad parameters received gradients.")
