"""Training loggers for obsweatherscale.

Provides a base :class:`TrainingLogger` interface and three concrete
implementations:

- :class:`TerminalLogger` — logs via Python's :mod:`logging` module.
- :class:`CSVLogger` — writes per-iteration metrics to a CSV file and
  hyperparameters to a JSON sidecar.
- :class:`MLflowLogger` — logs parameters and metrics to an MLflow
  tracking server.  Requires the optional ``mlflow`` dependency.
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TrainingLogger(ABC):
    """Abstract base class for training loggers.

    Subclasses must implement :meth:`log_params`, :meth:`log_metrics`,
    and :meth:`close`.
    """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log training hyperparameters and configuration.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of hyperparameter names and values.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a single training iteration.

        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names and values.
        step : int
            Current iteration number (1-based).
        """

    @abstractmethod
    def close(self) -> None:
        """Finalize and release any resources held by the logger."""


class TerminalLogger(TrainingLogger):
    """Logger that writes training progress via Python's :mod:`logging`.

    Parameters
    ----------
    name : str, default='obsweatherscale.training'
        Name of the Python logger instance.
    level : int, default=logging.INFO
        Logging level.
    """

    def __init__(
        self,
        name: str = "obsweatherscale.training",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(message)s")
            )
            self._logger.addHandler(handler)
        self._n_iter: int | None = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the terminal."""
        self._n_iter = params.get("n_iter")
        self._logger.info("Training parameters: %s", params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log per-iteration metrics to the terminal."""
        n_iter_str = f"/{self._n_iter}" if self._n_iter else ""
        self._logger.info(
            "Iter %d%s - train_loss: %.3f   val_loss: %.3f   "
            "train_time: %.3fs   iter_time: %.3fs",
            step,
            n_iter_str,
            metrics["train_loss"],
            metrics["val_loss"],
            metrics["train_time"],
            metrics["iter_time"],
        )

    def close(self) -> None:
        """No-op for terminal logging."""


class CSVLogger(TrainingLogger):
    """Logger that writes per-iteration metrics to a CSV file.

    Hyperparameters are stored in a JSON sidecar file with the same
    stem (e.g. ``log.csv`` → ``log.json``).

    Parameters
    ----------
    filepath : Path or str
        Path to the CSV output file.  Parent directories are created
        automatically.
    """

    def __init__(self, filepath: Path | str) -> None:
        self._filepath = Path(filepath)
        self._file: Any = None
        self._writer: Any = None
        self._header_written = False

    def log_params(self, params: dict[str, Any]) -> None:
        """Write hyperparameters to a JSON sidecar file."""
        params_path = self._filepath.with_suffix(".json")
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append one row of metrics to the CSV file."""
        if self._file is None:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(  # noqa: SIM115
                self._filepath, "w", newline="", encoding="utf-8"
            )
            self._writer = csv.writer(self._file)
        if not self._header_written:
            self._writer.writerow(["step", *metrics.keys()])
            self._header_written = True
        self._writer.writerow([step, *metrics.values()])

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            self._file.close()


class MLflowLogger(TrainingLogger):
    """Logger that records parameters and metrics to MLflow.

    Requires the optional ``mlflow`` package.  If no active MLflow run
    exists when the logger is constructed, a new run is started
    automatically and ended on :meth:`close`.

    Parameters
    ----------
    experiment_name : str, optional
        MLflow experiment name.  If provided,
        :func:`mlflow.set_experiment` is called.
    run_name : str, optional
        Name for the MLflow run (used only when a new run is started).
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        try:
            import mlflow  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. "
                "Install it with:  pip install mlflow"
            ) from exc

        self._mlflow = mlflow
        self._managed_run = False

        if experiment_name is not None:
            self._mlflow.set_experiment(experiment_name)

        if self._mlflow.active_run() is None:
            self._mlflow.start_run(run_name=run_name)
            self._managed_run = True

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active MLflow run."""
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log per-iteration metrics to the active MLflow run."""
        self._mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        """End the MLflow run if it was started by this logger."""
        if self._managed_run:
            self._mlflow.end_run()
