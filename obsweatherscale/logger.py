"""Loggers for obsweatherscale.

Provides a base :class:`Logger` interface and three concrete
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


class Logger(ABC):
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


class TerminalLogger(Logger):
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
        metrics_str = "   ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        self._logger.info("Iter %d%s - %s", step, n_iter_str, metrics_str)

    def close(self) -> None:
        """No-op for terminal logging."""


class CSVLogger(Logger):
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
        self._header_written = False

    def log_params(self, params: dict[str, Any]) -> None:
        """Write hyperparameters to a JSON sidecar file."""
        params_path = self._filepath.with_suffix(".json")
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append one row of metrics to the CSV file."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._header_written else "a"
        with open(self._filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["step", *metrics.keys()])
                self._header_written = True
            writer.writerow([step, *metrics.values()])

    def close(self) -> None:
        """No-op: the CSV file is opened and closed within each
        :meth:`log_metrics` call.
        """


class MLflowLogger(Logger):
    """Logger that records parameters and metrics to MLflow in
    optionally nested runs.

    Requires the optional ``mlflow`` package. If no active MLflow run
    exists when the logger is constructed, a new run is started
    automatically and ended on :meth:`close`.

    Modes
    -----
    Standard mode
        parent_run_name=None

        Behaves in a non-nested way:
        - uses the active run if one exists
        - otherwise creates a run named run_name

    Nested mode
        parent_run_name=<name>

        - if a run named parent_run_name is currently active, reuses it
          as parent run
        - otherwise, if a run named parent_run_name already exists in
          the experiment, the most recent one is reused (looked up by
          name, latest start time wins)
        - otherwise a new parent run named parent_run_name is created
        - creates/reuses a parent run named parent_run_name
        - a new child run named run_name is always created under the
          parent
        - all logging goes to the child run
        - on :meth:`close`, only runs that were started by this logger
          are ended; externally started runs are left open
    
    Notes
    -----
    Run names are not unique in MLflow. If multiple runs share the same
    parent_run_name, the most recently started one is used.
    
    Parameters
    ----------
    experiment_name : str, optional
        MLflow experiment name. If provided,
        :func:`mlflow.set_experiment` is called.
    run_name : str, optional
        Name for the MLflow run (used only when a new run is started).
    parent_run_name : str, optional
        Name for the parent MLflow run (used only in nested mode).
    run_tags : dict[str, str], optional
        Tags to set on the child (or only) run.
    parent_tags : dict[str, str], optional
        Tags to set on the parent run (only applied when a new parent
        is created).
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        parent_run_name: str | None = None,
        run_tags: dict[str, str] | None = None,
        parent_tags: dict[str, str] | None = None,
    ) -> None:
        try:
            import mlflow  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MLflowLogger. "
                "Install it with: pip install mlflow"
            ) from exc

        self._mlflow = mlflow
        self._managed_parent = False
        self._managed_child = False

        # Set and get active experiment
        if experiment_name is not None:
            self._mlflow.set_experiment(experiment_name)

        active_experiment = self._mlflow.get_experiment_by_name(
            experiment_name or "Default"
        )
        experiment_id = (
            active_experiment.experiment_id
            if active_experiment is not None else None
        )

        # Set run kwargs
        run_kwargs: dict[str, Any] = {"run_name": run_name}
        if run_tags is not None:
            run_kwargs["tags"] = run_tags

        parent_run_kwargs: dict[str, Any] = {"run_name": parent_run_name}
        if parent_tags is not None:
            parent_run_kwargs["tags"] = parent_tags

        # ---- Standard mode ----
        if parent_run_name is None:
            if self._mlflow.active_run() is None:
                self._mlflow.start_run(**run_kwargs)
                self._managed_child = True

        # ---- Nested mode ----
        else:
            active = self._mlflow.active_run()

            if active is not None:
                # Validate that the active run is the expected parent
                active_name = active.data.tags.get("mlflow.runName")
                if active_name != parent_run_name:
                    raise RuntimeError(
                        f"Active MLflow run '{active_name}' does not match "
                        f"requested parent run '{parent_run_name}'."
                    )
                parent_run_id = active.info.run_id

            else:
                # Search for an existing RUNNING parent run with this name
                parent_run_id = self._find_run_by_name(
                    parent_run_name, experiment_id
                )

                if parent_run_id is not None:
                    # Re-activate the parent so the child can nest under it
                    self._mlflow.start_run(run_id=parent_run_id)
                else:
                    # Create a fresh parent
                    self._mlflow.start_run(**parent_run_kwargs)
                    self._managed_parent = True

            self._mlflow.start_run(nested=True, **run_kwargs)
            self._managed_child = True

    def _find_run_by_name(
        self,
        run_name: str,
        experiment_id: str | None
    ) -> str | None:
        client = self._mlflow.MlflowClient()
        search_kwargs: dict = {
            "filter_string": f"attributes.run_name = '{run_name}'",
            "max_results": 1,
        }
        if experiment_id is not None:
            search_kwargs["experiment_ids"] = [experiment_id]

        results = client.search_runs(**search_kwargs)
        return results[0].info.run_id if results else None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active MLflow run."""
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log per-iteration metrics to the active MLflow run."""
        self._mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        """End MLflow run if it was started by this logger.

        If used in standard mode, this will end the run provided it was
        started by this logger.

        If used in nested mode, this will end the child run that was
        started, and the parent run if it was started by this logger.
        """
        if self._managed_child:  # end child run first
            self._mlflow.end_run()

        if self._managed_parent:
            self._mlflow.end_run()
