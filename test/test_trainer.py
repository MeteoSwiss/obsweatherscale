from pathlib import Path

import pytest
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean

import obsweatherscale as ows

from test.conftest import MyDataset


def make_trainer(dataset_train: MyDataset) -> ows.Trainer:
    train_x, train_y = dataset_train.get_dataset()
    model = ows.GPModel(
        LinearMean(3), RBFKernel(), GaussianLikelihood(), train_x, train_y,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return ows.Trainer(
        model,
        ows.make_mll_loss(model),
        ows.make_crps_loss(model),
        torch.device("cpu"),
        optimizer,
    )


def fit(
    trainer: ows.Trainer,
    datasets: tuple[MyDataset, MyDataset, MyDataset],
    **kwargs,
) -> ows.Trainer:
    return trainer.fit(
        *datasets, batch_size=2, n_iter=4, verbose=False, **kwargs,
    )


@pytest.fixture
def datasets(
    dataset_train: MyDataset,
    dataset_val_c: MyDataset,
    dataset_val_t: MyDataset,
) -> tuple[MyDataset, MyDataset, MyDataset]:
    return dataset_train, dataset_val_c, dataset_val_t


def test_iterations_are_one_based(
    datasets: tuple[MyDataset, MyDataset, MyDataset],
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "log.csv"
    trainer = fit(
        make_trainer(datasets[0]), datasets,
        loggers=[ows.CSVLogger(csv_path)],
    )

    assert [h["iter"] for h in trainer.history] == [1, 2, 3, 4]
    steps = [line.split(",")[0] for line in csv_path.read_text().splitlines()]
    assert steps[1:5] == ["1", "2", "3", "4"]


def test_no_checkpoints_by_default(
    datasets: tuple[MyDataset, MyDataset, MyDataset],
    tmp_path: Path,
) -> None:
    fit(make_trainer(datasets[0]), datasets, output_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["best_model.pt"]


def test_checkpoint_every(
    datasets: tuple[MyDataset, MyDataset, MyDataset],
    tmp_path: Path,
) -> None:
    fit(
        make_trainer(datasets[0]), datasets,
        output_dir=tmp_path, checkpoint_every=2,
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "best_model.pt", "model_2.pt", "model_4.pt",
    ]


def test_checkpoint_every_must_be_positive(
    datasets: tuple[MyDataset, MyDataset, MyDataset],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        fit(
            make_trainer(datasets[0]), datasets,
            output_dir=tmp_path, checkpoint_every=0,
        )
