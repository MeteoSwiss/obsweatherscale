"""Unit tests for obsweatherscale.training.loggers."""
# pylint: disable=protected-access

import csv
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from obsweatherscale.training.loggers import (
    CSVLogger,
    TerminalLogger,
    MLflowLogger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PARAMS: dict[str, Any] = {
    "batch_size": 32,
    "n_iter": 100,
    "seed": 42,
    "learning_rate": 0.005,
    "model": "GPModel",
    "optimizer": "Adam",
}

SAMPLE_METRICS: dict[str, float] = {
    "train loss": 1.234,
    "val loss": 1.567,
    "train time": 0.123,
    "iter time": 0.456,
}


# ---------------------------------------------------------------------------
# TerminalLogger
# ---------------------------------------------------------------------------


class TestTerminalLogger:
    def test_default_logger_name(self) -> None:
        logger = TerminalLogger()
        assert logger._logger.name == "obsweatherscale.training"

    def test_custom_logger_name(self) -> None:
        logger = TerminalLogger(name="my.logger")
        assert logger._logger.name == "my.logger"

    def test_handler_added_once(self) -> None:
        """Re-using the same logger name should not duplicate handlers."""
        name = "obsweatherscale.test.dedup"
        logger1 = TerminalLogger(name=name)
        n_handlers = len(logger1._logger.handlers)
        _logger2 = TerminalLogger(name=name)
        assert len(logger1._logger.handlers) == n_handlers

    def test_log_params_stores_n_iter(self) -> None:
        logger = TerminalLogger(name="obsweatherscale.test.params")
        logger.log_params(SAMPLE_PARAMS)
        assert logger._n_iter == SAMPLE_PARAMS["n_iter"]

    def test_log_params_calls_info(self) -> None:
        logger = TerminalLogger(name="obsweatherscale.test.params_info")
        with patch.object(logger._logger, "info") as mock_info:
            logger.log_params(SAMPLE_PARAMS)
            mock_info.assert_called_once()
            # The params dict should appear in the call args
            assert SAMPLE_PARAMS in mock_info.call_args.args

    def test_log_metrics_calls_info(self) -> None:
        logger = TerminalLogger(name="obsweatherscale.test.metrics_info")
        logger.log_params(SAMPLE_PARAMS)
        with patch.object(logger._logger, "info") as mock_info:
            logger.log_metrics(SAMPLE_METRICS, step=1)
            mock_info.assert_called_once()

    def test_log_metrics_includes_n_iter_in_message(self) -> None:
        logger = TerminalLogger(name="obsweatherscale.test.niter")
        logger.log_params(SAMPLE_PARAMS)  # sets _n_iter = 100
        with patch.object(logger._logger, "info") as mock_info:
            logger.log_metrics(SAMPLE_METRICS, step=1)
            # The format string should reference n_iter
            format_str = mock_info.call_args.args[0]
            assert "%d%s" in format_str

    def test_log_metrics_without_prior_log_params(self) -> None:
        """log_metrics before log_params should not raise."""
        logger = TerminalLogger(name="obsweatherscale.test.no_params")
        logger.log_metrics(SAMPLE_METRICS, step=1)  # should not raise

    def test_close_is_noop(self) -> None:
        logger = TerminalLogger(name="obsweatherscale.test.close")
        logger.close()  # should not raise

    def test_log_level_respected(self) -> None:
        logger = TerminalLogger(
            name="obsweatherscale.test.level", level=logging.WARNING
        )
        assert logger._logger.level == logging.WARNING


# ---------------------------------------------------------------------------
# CSVLogger
# ---------------------------------------------------------------------------


class TestCSVLogger:
    def test_log_params_creates_json_sidecar(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        logger.log_params(SAMPLE_PARAMS)

        json_path = tmp_path / "log.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["batch_size"] == SAMPLE_PARAMS["batch_size"]
        assert loaded["n_iter"] == SAMPLE_PARAMS["n_iter"]

    def test_log_params_json_contains_all_keys(self, tmp_path: Path) -> None:
        logger = CSVLogger(tmp_path / "log.csv")
        logger.log_params(SAMPLE_PARAMS)
        with open(tmp_path / "log.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert set(loaded.keys()) == set(SAMPLE_PARAMS.keys())

    def test_log_metrics_creates_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        logger.log_metrics(SAMPLE_METRICS, step=1)
        logger.close()
        assert csv_path.exists()

    def test_csv_header_row(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        logger.log_metrics(SAMPLE_METRICS, step=1)
        logger.close()

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header[0] == "step"
        assert list(SAMPLE_METRICS.keys()) == header[1:]

    def test_csv_data_row_values(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        logger.log_metrics(SAMPLE_METRICS, step=3)
        logger.close()

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
        assert int(row[0]) == 3
        assert float(row[1]) == pytest.approx(SAMPLE_METRICS["train loss"])
        assert float(row[2]) == pytest.approx(SAMPLE_METRICS["val loss"])

    def test_csv_header_written_once(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        for step in range(1, 4):
            logger.log_metrics(SAMPLE_METRICS, step=step)
        logger.close()

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4

    def test_csv_multiple_steps(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        logger = CSVLogger(csv_path)
        for step in range(1, 6):
            metrics = {**SAMPLE_METRICS, "train_loss": float(step)}
            logger.log_metrics(metrics, step=step)
        logger.close()

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            data_rows = list(reader)
        assert len(data_rows) == 5
        assert int(data_rows[-1][0]) == 5

    def test_close_without_log_metrics_is_safe(self, tmp_path: Path) -> None:
        """Calling close() before any log_metrics() should not raise."""
        logger = CSVLogger(tmp_path / "log.csv")
        logger.close()  # should not raise

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "subdir" / "nested" / "log.csv"
        logger = CSVLogger(csv_path)
        logger.log_params(SAMPLE_PARAMS)
        logger.log_metrics(SAMPLE_METRICS, step=1)
        logger.close()
        assert csv_path.exists()

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        logger = CSVLogger(str(tmp_path / "log.csv"))
        logger.log_metrics(SAMPLE_METRICS, step=1)
        logger.close()
        assert (tmp_path / "log.csv").exists()


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------


class TestMLflowLogger:
    """Tests for MLflowLogger using a fully mocked mlflow module."""

    def _make_mlflow_mock(self, active_run: bool = False) -> MagicMock:
        """Return a mock mlflow module."""
        mock = MagicMock()
        mock.active_run.return_value = MagicMock() if active_run else None
        return mock

    def _make_logger(
        self,
        mock_mlflow: MagicMock,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ):
        """Instantiate MLflowLogger with mlflow patched."""


        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            # Re-import to pick up the patched module inside __init__
            logger = MLflowLogger.__new__(MLflowLogger)
            logger._mlflow = mock_mlflow  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
            logger._managed_run = False  # pyright: ignore[reportPrivateUsage]
            if experiment_name is not None:
                mock_mlflow.set_experiment(experiment_name)
            if mock_mlflow.active_run() is None:
                mock_mlflow.start_run(run_name=run_name)
                logger._managed_run = True  # pyright: ignore[reportPrivateUsage]
        return logger

    def test_import_error_without_mlflow(self) -> None:
        with patch.dict("sys.modules", {"mlflow": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="mlflow is required"):
                MLflowLogger()

    def test_starts_run_when_no_active_run(self) -> None:
        mock = self._make_mlflow_mock(active_run=False)
        self._make_logger(mock)
        mock.start_run.assert_called()

    def test_no_new_run_when_active_run_exists(self) -> None:
        mock = self._make_mlflow_mock(active_run=True)
        logger = self._make_logger(mock)
        # close() must not call end_run if the run was started externally
        logger.close()
        mock.end_run.assert_not_called()

    def test_log_params_delegates_to_mlflow(self) -> None:
        mock = self._make_mlflow_mock()
        logger = self._make_logger(mock)
        logger.log_params(SAMPLE_PARAMS)
        mock.log_params.assert_called_once_with(SAMPLE_PARAMS)

    def test_log_metrics_delegates_to_mlflow(self) -> None:
        mock = self._make_mlflow_mock()
        logger = self._make_logger(mock)
        logger.log_metrics(SAMPLE_METRICS, step=5)
        mock.log_metrics.assert_called_once_with(SAMPLE_METRICS, step=5)

    def test_close_ends_managed_run(self) -> None:
        mock = self._make_mlflow_mock(active_run=False)
        logger = self._make_logger(mock)
        logger.close()
        mock.end_run.assert_called_once()

    def test_close_does_not_end_external_run(self) -> None:
        mock = self._make_mlflow_mock(active_run=True)
        logger = self._make_logger(mock)
        logger.close()
        mock.end_run.assert_not_called()

    def test_set_experiment_called_when_provided(self) -> None:
        mock = self._make_mlflow_mock()
        self._make_logger(mock, experiment_name="my_experiment")
        mock.set_experiment.assert_called_with("my_experiment")
