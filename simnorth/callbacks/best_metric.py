"""Callback that tracks and logs the best value of a monitored metric.

Reads the monitor in ``on_validation_end`` so the value is already in
``trainer.callback_metrics`` regardless of whether it was logged in
``validation_step`` or computed at epoch end (e.g. SimNorth's ``val_n_clusters``,
logged from the LightningModule's ``on_validation_epoch_end``). The running best
is logged as ``best_<monitor>`` to the active logger (e.g. MLflow), and
``self.best`` is read back by the Optuna driver to score a trial.
"""

from lightning.pytorch.callbacks import Callback


class BestMetricTracker(Callback):
    def __init__(self, monitor: str, mode: str = "min"):
        assert mode in {"min", "max"}
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < best
        return current > best

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = current.item()

        if self.best is None or self._is_better(current, self.best):
            self.best = current
            # self.log() is not allowed in on_validation_end; log via the logger.
            if trainer.logger is not None and trainer.is_global_zero:
                trainer.logger.log_metrics({f"best_{self.monitor}": self.best}, step=trainer.global_step)
