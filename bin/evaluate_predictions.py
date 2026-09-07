"""Evaluate a trained downscaling model's samples against test-set ground truth.

Finds prediction netcdf files written by bin/predict.py, aligns them against
the raw test-split predictand data, and reports bias/RMSE/correlation plus a
few diagnostic plots (mean maps, bias map, domain-mean timeseries).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr

from ml_downscaling_emulator.cordex_ml_data import open_raw_dataset_split_predictands

app = typer.Typer()


def find_prediction_files(workdir: Path, checkpoint: str, dataset: str, split: str):
    preds_root = workdir / "samples" / checkpoint / dataset
    pred_files = sorted(preds_root.glob(f"**/{split}/**/predictions-*.nc"))
    if len(pred_files) == 0:
        raise FileNotFoundError(
            f"No predictions-*.nc files found under {preds_root} for split '{split}'. "
            "Run bin/predict.py first."
        )
    return pred_files


@app.command()
def main(
    workdir: Path,
    variable: str = typer.Option(
        ..., help="Target variable name as it appears in the predictand dataset, e.g. target_pr or target_tas"
    ),
    checkpoint: str = typer.Option(...),
    dataset: str = "ANT_historical/ant_historical",
    split: str = "test",
    output_dir: Path = None,
):
    pred_files = find_prediction_files(workdir, checkpoint, dataset, split)
    typer.echo(f"Found {len(pred_files)} prediction file(s):")
    for f in pred_files:
        typer.echo(f"  {f}")

    pred_ds = xr.open_mfdataset(
        pred_files,
        combine="nested",
        concat_dim="sample",
        data_vars="minimal",
        preprocess=lambda ds: ds.expand_dims("sample"),
    )
    pred_var = f"pred_{variable}"
    pred_da = pred_ds[pred_var].mean(dim="sample")

    truth_ds = open_raw_dataset_split_predictands(dataset, split)
    truth_da = truth_ds[variable].sel(time=pred_da["time"])
    mask = truth_ds["padding_mask"]

    pred_da = pred_da.where(mask)
    truth_da = truth_da.where(mask)

    diff = pred_da - truth_da

    bias_map = diff.mean(dim="time", skipna=True)
    rmse_map = np.sqrt((diff**2).mean(dim="time", skipna=True))
    corr_map = xr.corr(pred_da, truth_da, dim="time")

    overall_bias = float(diff.mean(skipna=True))
    overall_rmse = float(np.sqrt((diff**2).mean(skipna=True)))
    overall_corr = float(corr_map.mean(skipna=True))

    typer.echo(f"\n=== {variable} on {split} split ({len(pred_files)} sample(s)) ===")
    typer.echo(f"Domain-mean bias (pred - truth): {overall_bias:.6g}")
    typer.echo(f"Domain-mean RMSE:                {overall_rmse:.6g}")
    typer.echo(f"Domain-mean time correlation:    {overall_corr:.4f}")

    if output_dir is None:
        output_dir = workdir / "samples" / checkpoint / dataset / split / "eval"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    truth_da.mean(dim="time", skipna=True).plot(ax=axes[0])
    axes[0].set_title(f"Mean truth {variable}")
    pred_da.mean(dim="time", skipna=True).plot(ax=axes[1])
    axes[1].set_title(f"Mean prediction {variable}")
    bias_map.plot(ax=axes[2], cmap="RdBu_r", center=0)
    axes[2].set_title("Mean bias (pred - truth)")
    rmse_map.plot(ax=axes[3])
    axes[3].set_title("RMSE")
    fig.tight_layout()
    maps_path = output_dir / f"{variable}_maps.png"
    fig.savefig(maps_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    truth_da.mean(dim=["grid_latitude", "grid_longitude"], skipna=True).plot(
        ax=ax, label="truth"
    )
    pred_da.mean(dim=["grid_latitude", "grid_longitude"], skipna=True).plot(
        ax=ax, label="prediction"
    )
    ax.set_title(f"Domain-mean {variable} timeseries")
    ax.legend()
    fig.tight_layout()
    ts_path = output_dir / f"{variable}_timeseries.png"
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)

    typer.echo(f"\nSaved plots to {maps_path} and {ts_path}")


if __name__ == "__main__":
    app()
