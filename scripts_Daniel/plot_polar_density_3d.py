#!/usr/bin/env python3
"""
3D visualization utilities for PolarMACE charge and spin densities.

This module is intentionally separate from `plot_polar_density.py`.
It does not modify the density-construction workflow; it only consumes
its outputs and renders 3D views using matplotlib.

Notes:
- `ase gui` is useful for atom coordinates, but not for volumetric scalar
  fields such as electron density.
- For interactive volume visualization, `.cube` files are still better
  opened in ChimeraX or VESTA.
- Inside Python, this module provides a pragmatic 3D plot based on
  thresholded scatter points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read
try:
    from scripts_Daniel.plot_polar_density import (
        DensityGrid,
        PolarDensityResult,
        compute_density_grid,
        load_polar_density_from_file,
        load_polar_density_result,
    )
except ImportError:
    from plot_polar_density import (  # type: ignore
        DensityGrid,
        PolarDensityResult,
        compute_density_grid,
        load_polar_density_from_file,
        load_polar_density_result,
    )


__all__ = [
    "Density3DSelection",
    "build_density_selection",
    "load_density_grid_from_npz",
    "plot_density_3d_scatter",
    "plot_density_3d_from_file",
    "plot_density_3d_from_result",
]


ATOM_COLORS = {
    1: "#f5f5f5",
    6: "#3c3c3c",
    7: "#2f67ff",
    8: "#d62828",
    9: "#4cc9f0",
    15: "#ff8800",
    16: "#ffd166",
    17: "#5dd39e",
}


@dataclass
class Density3DSelection:
    """Thresholded subsets of a density grid used for signed 3D plotting."""

    x_pos: np.ndarray
    y_pos: np.ndarray
    z_pos: np.ndarray
    values_pos: np.ndarray
    x_neg: np.ndarray
    y_neg: np.ndarray
    z_neg: np.ndarray
    values_neg: np.ndarray
    threshold: float
    field_name: str


def parse_args() -> argparse.Namespace:
    """CLI for quick 3D plots from either an existing NPZ or a structure/model pair."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None, help="Existing density NPZ created by plot_polar_density.py")
    parser.add_argument("--input", default=None, help="Structure readable by ASE")
    parser.add_argument("--model", default=None, help="Polar foundation model key or local checkpoint path")
    parser.add_argument("--index", default="0", help="ASE frame index")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", default="float32")
    parser.add_argument("--charge", type=float, default=None)
    parser.add_argument("--spin", type=float, default=None)
    parser.add_argument("--padding", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument(
        "--field",
        choices=["charge_density", "spin_density"],
        default="charge_density",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Keep only the largest absolute values above this quantile",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold override. If given, quantile is ignored.",
    )
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--output", default=None, help="Optional PNG output path")
    parser.add_argument(
        "--truncate-higher-l",
        action="store_true",
        help="Allow models with l_max > 1 and keep only monopole/dipole terms",
    )
    return parser.parse_args()


def load_density_grid_from_npz(npz_path: str | Path) -> DensityGrid:
    """Load a grid previously saved by `save_density_grid()`."""

    data = np.load(npz_path)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    return DensityGrid(
        x=x,
        y=y,
        z=z,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        charge_density=data["charge_density"],
        spin_density=data["spin_density"],
    )


def build_density_selection(
    density_grid: DensityGrid,
    field_name: str = "charge_density",
    threshold: float | None = None,
    quantile: float = 0.995,
    max_points: int = 30000,
) -> Density3DSelection:
    """Select signed high-density voxels for a readable 3D scatter plot.

    Matplotlib does not provide efficient volumetric visualization by default.
    This function keeps only grid points above a given absolute threshold and
    stores positive and negative regions separately, which behaves more like a
    signed isovalue rendering than a single mixed point cloud.
    """

    field = np.asarray(getattr(density_grid, field_name), dtype=np.float64)
    abs_field = np.abs(field)
    if threshold is None:
        threshold = float(np.quantile(abs_field, quantile))
    pos_mask = field >= threshold
    neg_mask = field <= -threshold

    x_pos = density_grid.grid_x[pos_mask]
    y_pos = density_grid.grid_y[pos_mask]
    z_pos = density_grid.grid_z[pos_mask]
    values_pos = field[pos_mask]

    x_neg = density_grid.grid_x[neg_mask]
    y_neg = density_grid.grid_y[neg_mask]
    z_neg = density_grid.grid_z[neg_mask]
    values_neg = field[neg_mask]

    max_points_per_sign = max(1, max_points // 2)
    if values_pos.size > max_points_per_sign:
        keep = np.argsort(np.abs(values_pos))[-max_points_per_sign:]
        x_pos = x_pos[keep]
        y_pos = y_pos[keep]
        z_pos = z_pos[keep]
        values_pos = values_pos[keep]
    if values_neg.size > max_points_per_sign:
        keep = np.argsort(np.abs(values_neg))[-max_points_per_sign:]
        x_neg = x_neg[keep]
        y_neg = y_neg[keep]
        z_neg = z_neg[keep]
        values_neg = values_neg[keep]

    return Density3DSelection(
        x_pos=x_pos,
        y_pos=y_pos,
        z_pos=z_pos,
        values_pos=values_pos,
        x_neg=x_neg,
        y_neg=y_neg,
        z_neg=z_neg,
        values_neg=values_neg,
        threshold=float(threshold),
        field_name=field_name,
    )


def _get_atom_color(atomic_number: int) -> str:
    return ATOM_COLORS.get(int(atomic_number), "#9aa0a6")


def plot_density_3d_scatter(
    atoms: Atoms,
    selection: Density3DSelection,
    *,
    point_size: float = 6.0,
    alpha: float = 0.18,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a 3D density view with matplotlib.

    Positive and negative regions are rendered as separate point clouds to avoid
    the visual mixing that happens when both signs are plotted in a single cloud.
    """

    fig = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    if selection.values_pos.size:
        ax.scatter(
            selection.x_pos,
            selection.y_pos,
            selection.z_pos,
            c="#d62828",
            s=point_size,
            alpha=alpha,
            linewidths=0.0,
            label=f"positive ({selection.values_pos.size})",
        )
    if selection.values_neg.size:
        ax.scatter(
            selection.x_neg,
            selection.y_neg,
            selection.z_neg,
            c="#2f67ff",
            s=point_size,
            alpha=alpha,
            linewidths=0.0,
            label=f"negative ({selection.values_neg.size})",
        )

    positions = atoms.get_positions()
    for atomic_number, pos in zip(atoms.numbers, positions):
        ax.scatter(
            [pos[0]],
            [pos[1]],
            [pos[2]],
            s=140,
            c=_get_atom_color(int(atomic_number)),
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )
        ax.text(
            pos[0],
            pos[1],
            pos[2],
            chemical_symbols[int(atomic_number)],
            fontsize=8,
        )

    ax.set_xlabel("x [A]")
    ax.set_ylabel("y [A]")
    ax.set_zlabel("z [A]")
    ax.set_box_aspect(
        (
            np.ptp(positions[:, 0]) + 2.0,
            np.ptp(positions[:, 1]) + 2.0,
            np.ptp(positions[:, 2]) + 2.0,
        )
    )
    ax.set_title(
        title
        or f"3D {selection.field_name} view, |rho| >= {selection.threshold:.3e}"
    )
    if selection.values_pos.size or selection.values_neg.size:
        ax.legend(loc="upper right")

    if output_path is not None:
        fig.savefig(output_path, dpi=220)
    if show:
        plt.show()
    return fig, ax


def plot_density_3d_from_result(
    density_result: PolarDensityResult,
    *,
    field_name: str = "charge_density",
    padding: float = 3.0,
    grid_size: int = 121,
    threshold: float | None = None,
    quantile: float = 0.995,
    max_points: int = 30000,
    point_size: float = 6.0,
    alpha: float = 0.18,
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Compute the grid and immediately render a 3D density plot."""

    density_grid = compute_density_grid(
        density_result=density_result,
        padding=padding,
        grid_size=grid_size,
    )
    selection = build_density_selection(
        density_grid=density_grid,
        field_name=field_name,
        threshold=threshold,
        quantile=quantile,
        max_points=max_points,
    )
    return plot_density_3d_scatter(
        atoms=density_result.atoms,
        selection=selection,
        point_size=point_size,
        alpha=alpha,
        output_path=output_path,
        show=show,
    )


def plot_density_3d_from_file(
    input_path: str | Path,
    model: str,
    *,
    index: str = "0",
    device: str = "cpu",
    default_dtype: str = "float32",
    charge: float | None = None,
    spin: float | None = None,
    field_name: str = "charge_density",
    padding: float = 3.0,
    grid_size: int = 121,
    threshold: float | None = None,
    quantile: float = 0.995,
    max_points: int = 30000,
    point_size: float = 6.0,
    alpha: float = 0.18,
    output_path: str | Path | None = None,
    show: bool = True,
    truncate_higher_l: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Full convenience wrapper from structure file to 3D matplotlib plot."""

    density_result = load_polar_density_from_file(
        input_path=input_path,
        model=model,
        index=index,
        device=device,
        default_dtype=default_dtype,
        charge=charge,
        spin=spin,
        truncate_higher_l=truncate_higher_l,
    )
    return plot_density_3d_from_result(
        density_result=density_result,
        field_name=field_name,
        padding=padding,
        grid_size=grid_size,
        threshold=threshold,
        quantile=quantile,
        max_points=max_points,
        point_size=point_size,
        alpha=alpha,
        output_path=output_path,
        show=show,
    )


def main() -> None:
    """CLI entry point for quick 3D visualizations."""

    args = parse_args()
    if args.npz is None and (args.input is None or args.model is None):
        raise ValueError("Give either --npz or the pair --input/--model.")

    if args.npz is not None:
        density_grid = load_density_grid_from_npz(args.npz)
        if args.input is None:
            raise ValueError("When using --npz, also provide --input to load atoms for plotting.")
        atoms = read(args.input, index=args.index)
        selection = build_density_selection(
            density_grid=density_grid,
            field_name=args.field,
            threshold=args.threshold,
            quantile=args.quantile,
            max_points=args.max_points,
        )
        plot_density_3d_scatter(
            atoms=atoms,
            selection=selection,
            point_size=args.point_size,
            alpha=args.alpha,
            output_path=args.output,
            show=True,
        )
        return

    plot_density_3d_from_file(
        input_path=args.input,
        model=args.model,
        index=args.index,
        device=args.device,
        default_dtype=args.default_dtype,
        charge=args.charge,
        spin=args.spin,
        field_name=args.field,
        padding=args.padding,
        grid_size=args.grid_size,
        threshold=args.threshold,
        quantile=args.quantile,
        max_points=args.max_points,
        point_size=args.point_size,
        alpha=args.alpha,
        output_path=args.output,
        show=True,
        truncate_higher_l=args.truncate_higher_l,
    )


if __name__ == "__main__":
    main()
