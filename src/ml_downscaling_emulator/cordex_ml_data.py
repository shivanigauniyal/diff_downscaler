"""Loading CORDEX ML data into PyTorch"""

import logging
import cf_xarray  # noqa: F401
import gc
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from mlde_utils.transforms import build_input_transform, build_target_transform
from ml_downscaling_emulator import local_transforms  # noqa: F401  registers "clip0" transform

DATA_PATH = Path(os.getenv("DATA_PATH"))
# Zarr stores live under mlde-data/data/{dataset_name}/{split}/
DATASETS_PATH = DATA_PATH / "mlde-data" / "data"

logger = logging.getLogger(__name__)


def custom_collate(batch):
    from torch.utils.data import default_collate

    return *default_collate([tuple(e[:-1]) for e in batch]), np.concatenate(
        [e[-1] for e in batch]
    )


def get_variables(config):
    predictor_variables = list(config.data.predictor_variables)
    target_variables = config.data.target_variables
    static_variables = config.data.static_variables

    return predictor_variables, static_variables, target_variables


def _experiment_path(dataset_name, split):
    split_dir = split
    if split == "train+val":
        split_dir = "train"

    return DATASETS_PATH / dataset_name / split_dir


def _open_zarr_split(zarr_path):
    ds = xr.open_zarr(zarr_path)
    # Zarr stores have an ensemble_member dim; squeeze it out
    if "ensemble_member" in ds.dims:
        ds = ds.squeeze("ensemble_member", drop=True)
    if "axis" not in ds["time"].attrs:
        ds["time"].attrs["axis"] = "T"
    return ds


def open_raw_dataset_split_predictands(
    dataset_name,
    split,
):
    experiment_path = _experiment_path(dataset_name, split)
    return _open_zarr_split(experiment_path / "predictands.zarr")


def open_raw_dataset_split_predictors(
    dataset_name,
    split,
):
    experiment_path = _experiment_path(dataset_name, split)
    return _open_zarr_split(experiment_path / "predictors.zarr")


def open_raw_dataset_split_static_inputs(
    dataset_name,
    split,
):
    # Static fields are time-invariant; always read from the train directory
    static_path = (_experiment_path(dataset_name, "train")
                   / "predictors" / "Static_fields.nc")
    return xr.open_dataset(static_path)


def get_predictor_transform(
    dataset_name,
    key,
    variables,
    transform_dir,
):
    logger.debug("Opening training predictor dataset for input transform fitting")
    ds = open_raw_dataset_split_predictors(
        dataset_name,
        "train",
    )

    logger.debug("Building input transform object")
    input_transform = build_input_transform(variables, key)

    logger.debug("Fitting input transform")
    input_transform.fit(ds, ds)

    logger.debug("Memory cleanup after input transform fitting")
    ds.close()
    del ds
    gc.collect()

    return input_transform


def get_target_transform(
    dataset_name,
    keys,
    variables,
    transform_dir,
):
    logger.debug("Opening training predictand dataset for target transform fitting")
    ds = open_raw_dataset_split_predictands(
        dataset_name,
        "train",
    )

    logger.debug("Building target transform object")
    target_transform = build_target_transform(variables, keys)

    logger.debug("Fitting target transform")
    target_transform.fit(ds, ds)

    logger.debug("Memory cleanup after target transform fitting")
    ds.close()
    del ds
    gc.collect()

    return target_transform


def get_dataloader(
    dataset_name,
    predictor_variables,
    static_variables,
    target_variables,
    transform,
    target_transform,
    split,
    batch_size,
    shuffle=True,
    training=True,
    drop_last=False,
):
    predictor_ds = open_raw_dataset_split_predictors(
        dataset_name,
        split,
    )

    static_ds = open_raw_dataset_split_static_inputs(
        dataset_name,
        split,
    )

    predictor_ds = transform.transform(predictor_ds)

    if training:
        predictand_ds = open_raw_dataset_split_predictands(
            dataset_name,
            split,
        )
        predictand_ds = target_transform.transform(predictand_ds)

        pt_dataset = CordexMLTrainingDataset(
            predictor_ds,
            static_ds,
            predictand_ds,
            predictor_variables,
            static_variables,
            target_variables,
        )
    else:
        pt_dataset = CordexMLDataset(
            predictor_ds, static_ds, predictor_variables, static_variables
        )

    num_workers = min(8, os.cpu_count() or 1)
    data_loader = DataLoader(
        pt_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=drop_last,
    )

    return data_loader


class CordexMLDataset(Dataset):
    def __init__(self, predictor_ds, static_ds, variables, static_variables):
        self.variables = list(variables)
        self.static_variables = list(static_variables)

        self.predictor_da = predictor_ds[self.variables].cf.transpose("T", "Y", "X")
        self.static_da = static_ds.cf.transpose("Y", "X")[self.static_variables]
        # static fields (e.g. orography) aren't standardised upstream like the
        # predictor/target variables are, so scale each to roughly unit range
        for var in self.static_variables:
            scale = float(np.abs(self.static_da[var].values).max())
            if scale > 0:
                self.static_da[var] = self.static_da[var] / scale

    def __len__(self):
        return len(self.predictor_da.time)

    def __getitem__(self, idx):
        predictors = torch.tensor(
            # stack features before lat-lon (HW)
            np.stack(
                [self.predictor_da[var].isel(time=idx) for var in self.variables],
                axis=-3,
            ),
            dtype=torch.float32,
        )

        if len(self.static_variables) > 0:
            statics = torch.tensor(
                # stack features before lat-lon (HW)
                np.stack(
                    [self.static_da[var] for var in self.static_variables],
                    axis=-3,
                ),
                dtype=torch.float32,
            )
        else:
            statics = torch.tensor([])

        time = self.predictor_da.isel(time=idx)["time"].values.reshape(-1)

        return predictors, statics, time


class CordexMLTrainingDataset(CordexMLDataset):
    def __init__(
        self,
        predictor_ds,
        static_ds,
        predictand_ds,
        variables,
        static_variables,
        target_variables,
    ):
        super().__init__(predictor_ds, static_ds, variables, static_variables)

        self.target_variables = list(target_variables)

        self.predictand_da = predictand_ds[self.target_variables].cf.transpose(
            "T", "Y", "X"
        )

    def __getitem__(self, idx):
        predictors, statics, time = super().__getitem__(idx)

        predictands = torch.tensor(
            # stack features before lat-lon (HW)
            np.stack(
                [
                    self.predictand_da[var].isel(time=idx)
                    for var in self.target_variables
                ],
                axis=-3,
            ),
            dtype=torch.float32,
        )

        return predictors, statics, predictands, time


def np_samples_to_xr(
    np_samples, target_transform, target_vars, dims, var_attrs, coords, cf_data_vars
):
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any transformation applied to the target variables before modelling.
    """
    coords = {**dict(coords)}

    data_vars = {**cf_data_vars}
    for var_idx, var in enumerate(target_vars):
        np_var_pred = np_samples[:, var_idx, :]
        pred_var = (dims, np_var_pred, var_attrs[var])
        raw_pred_var = (
            dims,
            np_var_pred,
            {},
        )
        data_vars.update(
            {
                var: pred_var,  # don't rename pred var until after inverting target transform
                f"raw_pred_{var}": raw_pred_var,
            }
        )

    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    ).rename({var: f"pred_{var}" for var in target_vars})

    # Re-assign attributes as target_transform inversion removes them
    for var_idx, var in enumerate(target_vars):
        samples_ds[f"pred_{var}"] = samples_ds[f"pred_{var}"].assign_attrs(
            var_attrs[var]
        )
    return samples_ds
