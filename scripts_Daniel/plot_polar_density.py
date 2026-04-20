#!/usr/bin/env python3
"""
Plot charge-density and spin-density slices from a PolarMACE model.

The MACE-POLAR-1 paper expands the smooth spin-charge density as

    rho(r) = sum_{i,l,m} p_{i,lm} phi_{lm}(r - R_i)

where the p_{i,lm} are atom-centred multipole coefficients. In this
repository those coefficients are exposed as:

    - calculator.results["density_coefficients"]
    - calculator.results["spin_charge_density"]

This script reconstructs a smooth real-space field from those outputs.
For l_max <= 1 it preserves the atomic monopoles and dipoles exactly by
using a Cartesian Gaussian basis:

    g(r) = N exp(-|r|^2 / (2 sigma^2))
    rho_i(r) = q_i g(r) + mu_i . r / sigma^2 * g(r)

The foundation models described in the paper use l_max = 1, so this is
the relevant case for MACE-POLAR-1-M/L.

Typical library usage:

    from ase.io import read
    from plot_polar_density import (
        load_polar_density_result,
        compute_density_grid,
        extract_density_slice,
        save_density_cube,
    )

    atoms = read("molecule.xyz")
    result = load_polar_density_result(atoms, model="polar-1-m")
    grid = compute_density_grid(result, padding=3.0, grid_size=121)
    charge_slice = extract_density_slice(grid, field_name="charge_density", plane="xy")
    save_density_cube("charge_density.cube", result.atoms, grid, "charge_density")

Main data flow:
1. Run PolarMACE and read `density_coefficients` / `spin_charge_density`.
2. Reconstruct smooth charge and spin densities on a regular 3D grid.
3. Export the grid to `npz`, `cube`, or 2D `png` slices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import mace_polar


PLANE_TO_AXES = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}

__all__ = [
    "DensityGrid",
    "DensitySlice",
    "PolarDensityResult",
    "build_grid",
    "coeffs_to_dipoles",
    "compute_density_grid",
    "extract_density_slice",
    "get_default_slice_value",
    "load_polar_density_from_file",
    "load_polar_density_result",
    "plot_density_slice",
    "prepare_atoms",
    "reconstruct_density",
    "save_density_cube",
    "save_density_grid",
]


ANGSTROM_TO_BOHR = 1.8897259886


@dataclass
class PolarDensityResult:
    """Raw outputs needed to reconstruct real-space densities.

    Attributes:
        atoms: Structure used in the PolarMACE evaluation.
        charge_coefficients: Final charge-density multipoles p_{i,lm}.
        spin_charge_density: Spin-resolved multipoles with shape [n_atoms, 2, n_coeffs].
        spin_coefficients: Spin density coefficients, computed as up minus down.
        sigma: Gaussian smearing width used by the model for the density basis.
        l_max: Maximum multipole order available in the checkpoint.
    """

    atoms: Atoms
    charge_coefficients: np.ndarray
    spin_charge_density: np.ndarray
    spin_coefficients: np.ndarray
    sigma: float
    l_max: int


@dataclass
class DensityGrid:
    """Regular 3D grid and the reconstructed scalar fields on that grid."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    charge_density: np.ndarray
    spin_density: np.ndarray


@dataclass
class DensitySlice:
    """2D slice extracted from a 3D density field."""

    coord_a: np.ndarray
    coord_b: np.ndarray
    values: np.ndarray
    plane: str
    slice_value: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the standalone CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Structure readable by ASE")
    parser.add_argument(
        "--model",
        required=True,
        help="Polar foundation model key or local checkpoint path",
    )
    parser.add_argument("--index", default="0", help="ASE frame index")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-dtype", default="float32")
    parser.add_argument("--output-dir", default="polar_density_plots")
    parser.add_argument(
        "--charge",
        type=float,
        default=None,
        help="Override atoms.info['charge']",
    )
    parser.add_argument(
        "--spin",
        type=float,
        default=None,
        help="Override atoms.info['spin']. Closed-shell default is 1 in PolarMACE",
    )
    parser.add_argument("--padding", type=float, default=3.0)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=121,
        help="Number of points per axis for the 3D grid",
    )
    parser.add_argument(
        "--plane",
        choices=sorted(PLANE_TO_AXES),
        default="xy",
        help="Plane used for the plotted slice",
    )
    parser.add_argument(
        "--slice-value",
        type=float,
        default=None,
        help="Coordinate of the slice along the orthogonal axis in Angstrom",
    )
    parser.add_argument(
        "--truncate-higher-l",
        action="store_true",
        help="Allow models with l_max > 1 and keep only monopole/dipole terms",
    )
    parser.add_argument(
        "--no-cube",
        action="store_true",
        help="Disable Gaussian cube export",
    )
    return parser.parse_args()


def prepare_atoms(
    atoms: Atoms,
    charge: float | None = None,
    spin: float | None = None,
) -> Atoms:
    """Return a copy of `atoms` with PolarMACE-compatible charge/spin metadata.

    PolarMACE reads total charge from `atoms.info["charge"]` and total spin
    from `atoms.info["spin"]`. For closed-shell systems the code uses `spin=1.0`
    by default, mirroring the calculator conventions used by the repository.
    """

    atoms = atoms.copy()
    if charge is not None:
        atoms.info["charge"] = charge
    else:
        atoms.info.setdefault("charge", 0.0)
    if spin is not None:
        atoms.info["spin"] = spin
    else:
        atoms.info.setdefault("spin", 1.0)
    return atoms


def gaussian_s(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate the normalized s-type Gaussian basis function on a grid."""

    r2 = dx * dx + dy * dy + dz * dz
    prefactor = 1.0 / (((2.0 * np.pi) ** 1.5) * sigma**3)
    return prefactor * np.exp(-0.5 * r2 / sigma**2)


def coeffs_to_dipoles(coefficients: np.ndarray) -> np.ndarray:
    """Convert the first l=1 coefficients into Cartesian dipole components.

    The model stores multipoles in the same order used internally for
    `compute_total_charge_dipole_permuted()`. Reordering with `[2, 0, 1]`
    yields Cartesian `(x, y, z)` dipoles consistent with that utility.
    """

    if coefficients.shape[1] < 4:
        return np.zeros((coefficients.shape[0], 3), dtype=coefficients.dtype)
    # The internal order is converted to Cartesian xyz in compute_total_charge_dipole_permuted().
    return coefficients[:, 1:4][:, [2, 0, 1]]


def load_polar_density_result(
    atoms: Atoms,
    model: str,
    device: str = "cpu",
    default_dtype: str = "float32",
    charge: float | None = None,
    spin: float | None = None,
    truncate_higher_l: bool = False,
) -> PolarDensityResult:
    """Run a PolarMACE model on an ASE Atoms object and collect density outputs.

    This is the main entry point when using the module as a library.
    """

    atoms = prepare_atoms(atoms=atoms, charge=charge, spin=spin)
    calculator = mace_polar(
        model=model,
        device=device,
        default_dtype=default_dtype,
    )
    atoms.calc = calculator
    atoms.get_potential_energy()

    raw_model = calculator.models[0]
    l_max = int(getattr(raw_model, "atomic_multipoles_max_l", 0))
    sigma = float(getattr(raw_model, "atomic_multipoles_smearing_width", 1.0))
    if l_max > 1 and not truncate_higher_l:
        raise ValueError(
            f"This script reconstructs l<=1 exactly. The loaded model has l_max={l_max}. "
            "Pass truncate_higher_l=True to keep only monopole/dipole terms."
        )

    charge_coeffs = np.asarray(
        calculator.results["density_coefficients"], dtype=np.float64
    )
    spin_channels = np.asarray(
        calculator.results["spin_charge_density"], dtype=np.float64
    )
    if l_max > 1:
        charge_coeffs = charge_coeffs[:, :4]
        spin_channels = spin_channels[:, :, :4]

    spin_coeffs = spin_channels[:, 0, :] - spin_channels[:, 1, :]
    return PolarDensityResult(
        atoms=atoms,
        charge_coefficients=charge_coeffs,
        spin_charge_density=spin_channels,
        spin_coefficients=spin_coeffs,
        sigma=sigma,
        l_max=l_max,
    )


def load_polar_density_from_file(
    input_path: str | Path,
    model: str,
    index: str = "0",
    device: str = "cpu",
    default_dtype: str = "float32",
    charge: float | None = None,
    spin: float | None = None,
    truncate_higher_l: bool = False,
) -> PolarDensityResult:
    """Read a structure with ASE and forward it to `load_polar_density_result()`."""

    atoms = read(input_path, index=index)
    return load_polar_density_result(
        atoms=atoms,
        model=model,
        device=device,
        default_dtype=default_dtype,
        charge=charge,
        spin=spin,
        truncate_higher_l=truncate_higher_l,
    )


def reconstruct_density(
    positions: np.ndarray,
    coefficients: np.ndarray,
    sigma: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    """Reconstruct a smooth scalar field from monopoles and dipoles.

    For `l_max <= 1` the paper's Gaussian multipole expansion can be written
    as a sum of one s-type Gaussian plus its first derivatives. This preserves
    the atomic monopole and dipole moments exactly on the continuous field.
    """

    field = np.zeros_like(grid_x, dtype=np.float64)
    monopoles = coefficients[:, 0]
    dipoles = coeffs_to_dipoles(coefficients)

    for idx, center in enumerate(positions):
        # Shift the full grid so each atom contributes from its own local origin.
        dx = grid_x - center[0]
        dy = grid_y - center[1]
        dz = grid_z - center[2]
        g0 = gaussian_s(dx, dy, dz, sigma)
        field += monopoles[idx] * g0
        field += (dipoles[idx, 0] * dx + dipoles[idx, 1] * dy + dipoles[idx, 2] * dz) * (
            g0 / sigma**2
        )
    return field


def build_grid(
    positions: np.ndarray, padding: float, grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a Cartesian grid enclosing the structure with uniform padding."""

    lower = positions.min(axis=0) - padding
    upper = positions.max(axis=0) + padding
    xs = np.linspace(lower[0], upper[0], grid_size)
    ys = np.linspace(lower[1], upper[1], grid_size)
    zs = np.linspace(lower[2], upper[2], grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    return xs, ys, zs, grid_x, grid_y, grid_z


def compute_density_grid(
    density_result: PolarDensityResult,
    padding: float = 3.0,
    grid_size: int = 121,
) -> DensityGrid:
    """Reconstruct charge and spin densities on a regular 3D grid."""

    positions = density_result.atoms.get_positions()
    xs, ys, zs, grid_x, grid_y, grid_z = build_grid(
        positions=positions,
        padding=padding,
        grid_size=grid_size,
    )
    charge_density = reconstruct_density(
        positions=positions,
        coefficients=density_result.charge_coefficients,
        sigma=density_result.sigma,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
    )
    spin_density = reconstruct_density(
        positions=positions,
        coefficients=density_result.spin_coefficients,
        sigma=density_result.sigma,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
    )
    return DensityGrid(
        x=xs,
        y=ys,
        z=zs,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        charge_density=charge_density,
        spin_density=spin_density,
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    """Return the index of the grid point closest to `target`."""

    return int(np.argmin(np.abs(values - target)))


def extract_slice(
    field: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    plane: str,
    slice_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2D plane from a 3D field defined on `(x, y, z)` coordinates."""

    xs, ys, zs = coords
    axis_a, axis_b, axis_fixed = PLANE_TO_AXES[plane]
    axes = [xs, ys, zs]
    fixed_values = axes[axis_fixed]
    idx = nearest_index(fixed_values, slice_value)

    if axis_fixed == 0:
        slice_2d = field[idx, :, :]
    elif axis_fixed == 1:
        slice_2d = field[:, idx, :]
    else:
        slice_2d = field[:, :, idx]

    coord_a, coord_b = np.meshgrid(axes[axis_a], axes[axis_b], indexing="ij")
    return coord_a, coord_b, slice_2d


def get_default_slice_value(positions: np.ndarray, coords: tuple[np.ndarray, np.ndarray, np.ndarray], plane: str) -> float:
    """Choose a default slice that passes near the molecular center."""

    _, _, axis_fixed = PLANE_TO_AXES[plane]
    fixed_coords = [coords[0], coords[1], coords[2]][axis_fixed]
    default_slice = float(np.mean(positions[:, axis_fixed]))
    return float(fixed_coords[nearest_index(fixed_coords, default_slice)])


def extract_density_slice(
    density_grid: DensityGrid,
    field_name: str,
    plane: str = "xy",
    slice_value: float | None = None,
) -> DensitySlice:
    """Extract a named scalar field from `DensityGrid` as a 2D slice."""

    field = getattr(density_grid, field_name)
    coords = (density_grid.x, density_grid.y, density_grid.z)
    if slice_value is None:
        _, _, axis_fixed = PLANE_TO_AXES[plane]
        center = 0.5 * (coords[axis_fixed][0] + coords[axis_fixed][-1])
        slice_value = float(coords[axis_fixed][nearest_index(coords[axis_fixed], center)])
    coord_a, coord_b, values = extract_slice(field, coords, plane, float(slice_value))
    return DensitySlice(
        coord_a=coord_a,
        coord_b=coord_b,
        values=values,
        plane=plane,
        slice_value=float(slice_value),
    )


def plot_slice(
    output_path: Path,
    atoms_positions: np.ndarray,
    values_2d: np.ndarray,
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    plane: str,
    slice_axis_value: float,
    title: str,
    cmap: str,
    symmetric: bool,
) -> None:
    """Plot a 2D scalar field slice and overlay atoms near the slicing plane."""

    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)

    extent = [
        float(coord_a.min()),
        float(coord_a.max()),
        float(coord_b.min()),
        float(coord_b.max()),
    ]
    image = values_2d.T
    kwargs = {"origin": "lower", "extent": extent, "cmap": cmap, "aspect": "equal"}
    if symmetric:
        vmax = float(np.max(np.abs(values_2d)))
        kwargs["vmin"] = -vmax
        kwargs["vmax"] = vmax
    im = ax.imshow(image, **kwargs)
    fig.colorbar(im, ax=ax, shrink=0.9)

    axis_a, axis_b, axis_fixed = PLANE_TO_AXES[plane]
    labels = ["x", "y", "z"]
    tolerance = 0.25
    mask = np.abs(atoms_positions[:, axis_fixed] - slice_axis_value) <= tolerance
    shown = atoms_positions[mask]
    ax.scatter(
        shown[:, axis_a],
        shown[:, axis_b],
        c="white",
        edgecolors="black",
        s=40,
        linewidths=0.8,
    )
    ax.set_xlabel(f"{labels[axis_a]} [A]")
    ax.set_ylabel(f"{labels[axis_b]} [A]")
    ax.set_title(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_density_slice(
    output_path: str | Path,
    atoms: Atoms,
    density_slice: DensitySlice,
    title: str,
    cmap: str = "coolwarm",
    symmetric: bool = True,
) -> None:
    """Convenience wrapper around `plot_slice()` using `DensitySlice`."""

    plot_slice(
        output_path=Path(output_path),
        atoms_positions=atoms.get_positions(),
        values_2d=density_slice.values,
        coord_a=density_slice.coord_a,
        coord_b=density_slice.coord_b,
        plane=density_slice.plane,
        slice_axis_value=density_slice.slice_value,
        title=title,
        cmap=cmap,
        symmetric=symmetric,
    )


def save_density_grid(
    output_path: str | Path,
    density_result: PolarDensityResult,
    density_grid: DensityGrid,
) -> None:
    """Save the reconstructed fields and metadata to a compressed NumPy archive."""

    np.savez_compressed(
        output_path,
        x=density_grid.x,
        y=density_grid.y,
        z=density_grid.z,
        charge_density=density_grid.charge_density,
        spin_density=density_grid.spin_density,
        charge_coefficients=density_result.charge_coefficients,
        spin_charge_density=density_result.spin_charge_density,
        sigma=density_result.sigma,
        l_max=density_result.l_max,
        charge=float(density_result.atoms.info["charge"]),
        spin=float(density_result.atoms.info["spin"]),
    )


def save_density_cube(
    output_path: str | Path,
    atoms: Atoms,
    density_grid: DensityGrid,
    field_name: str,
    comment: str | None = None,
) -> None:
    """Export one reconstructed field to Gaussian cube format.

    The cube format stores:
    - the atom list
    - the grid origin and spacing
    - one scalar value per voxel

    Most molecular viewers read coordinates in Bohr inside cube files, so the
    function converts the grid and atom positions from Angstrom before writing.
    """

    field = np.asarray(getattr(density_grid, field_name), dtype=np.float64)
    output_path = Path(output_path)

    nx, ny, nz = field.shape
    x0 = float(density_grid.x[0])
    y0 = float(density_grid.y[0])
    z0 = float(density_grid.z[0])
    dx = float(density_grid.x[1] - density_grid.x[0]) if nx > 1 else 1.0
    dy = float(density_grid.y[1] - density_grid.y[0]) if ny > 1 else 1.0
    dz = float(density_grid.z[1] - density_grid.z[0]) if nz > 1 else 1.0

    origin_bohr = np.array([x0, y0, z0]) * ANGSTROM_TO_BOHR
    step_x_bohr = np.array([dx, 0.0, 0.0]) * ANGSTROM_TO_BOHR
    step_y_bohr = np.array([0.0, dy, 0.0]) * ANGSTROM_TO_BOHR
    step_z_bohr = np.array([0.0, 0.0, dz]) * ANGSTROM_TO_BOHR

    with output_path.open("w", encoding="ascii") as handle:
        handle.write(f"{comment or f'PolarMACE {field_name}'}\n")
        handle.write("Generated by plot_polar_density.py\n")
        handle.write(
            f"{len(atoms):5d} {origin_bohr[0]:11.6f} {origin_bohr[1]:11.6f} {origin_bohr[2]:11.6f}\n"
        )
        handle.write(
            f"{nx:5d} {step_x_bohr[0]:11.6f} {step_x_bohr[1]:11.6f} {step_x_bohr[2]:11.6f}\n"
        )
        handle.write(
            f"{ny:5d} {step_y_bohr[0]:11.6f} {step_y_bohr[1]:11.6f} {step_y_bohr[2]:11.6f}\n"
        )
        handle.write(
            f"{nz:5d} {step_z_bohr[0]:11.6f} {step_z_bohr[1]:11.6f} {step_z_bohr[2]:11.6f}\n"
        )

        positions_bohr = atoms.get_positions() * ANGSTROM_TO_BOHR
        for atomic_number, position in zip(atoms.numbers, positions_bohr):
            handle.write(
                f"{int(atomic_number):5d} {0.0:11.6f} {position[0]:11.6f} {position[1]:11.6f} {position[2]:11.6f}\n"
            )

        values_per_line = 6
        # Cube files are written as a flat stream ordered by x, then y, then z.
        for ix in range(nx):
            for iy in range(ny):
                line_values = []
                for iz in range(nz):
                    line_values.append(f"{field[ix, iy, iz]:13.5e}")
                    if len(line_values) == values_per_line:
                        handle.write("".join(line_values) + "\n")
                        line_values = []
                if line_values:
                    handle.write("".join(line_values) + "\n")


def main() -> None:
    """CLI entry point: run the full workflow from structure to output files."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_result = load_polar_density_from_file(
        input_path=args.input,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        index=args.index,
        charge=args.charge,
        spin=args.spin,
        truncate_higher_l=args.truncate_higher_l,
    )
    density_grid = compute_density_grid(
        density_result=density_result,
        padding=args.padding,
        grid_size=args.grid_size,
    )
    positions = density_result.atoms.get_positions()
    coords = (density_grid.x, density_grid.y, density_grid.z)
    if args.slice_value is None:
        slice_value = get_default_slice_value(positions, coords, args.plane)
    else:
        _, _, axis_fixed = PLANE_TO_AXES[args.plane]
        fixed_coords = coords[axis_fixed]
        slice_value = float(fixed_coords[nearest_index(fixed_coords, float(args.slice_value))])
    charge_slice = extract_density_slice(
        density_grid=density_grid,
        field_name="charge_density",
        plane=args.plane,
        slice_value=slice_value,
    )
    spin_slice = extract_density_slice(
        density_grid=density_grid,
        field_name="spin_density",
        plane=args.plane,
        slice_value=slice_value,
    )

    stem = Path(args.input).stem
    save_density_grid(
        output_path=output_dir / f"{stem}_polar_density_grid.npz",
        density_result=density_result,
        density_grid=density_grid,
    )
    if not args.no_cube:
        save_density_cube(
            output_path=output_dir / f"{stem}_charge_density.cube",
            atoms=density_result.atoms,
            density_grid=density_grid,
            field_name="charge_density",
            comment="PolarMACE smooth charge density",
        )
        save_density_cube(
            output_path=output_dir / f"{stem}_spin_density.cube",
            atoms=density_result.atoms,
            density_grid=density_grid,
            field_name="spin_density",
            comment="PolarMACE smooth spin density",
        )

    plot_density_slice(
        output_path=output_dir / f"{stem}_charge_density_{args.plane}.png",
        atoms=density_result.atoms,
        density_slice=charge_slice,
        title=f"PolarMACE smooth charge density ({args.plane} @ {slice_value:.2f} A)",
        cmap="coolwarm",
        symmetric=True,
    )
    plot_density_slice(
        output_path=output_dir / f"{stem}_spin_density_{args.plane}.png",
        atoms=density_result.atoms,
        density_slice=spin_slice,
        title=f"PolarMACE smooth spin density ({args.plane} @ {slice_value:.2f} A)",
        cmap="seismic",
        symmetric=True,
    )

    print(f"Saved outputs in {output_dir}")
    print(f"  sigma = {density_result.sigma:.6f} A")
    print(f"  l_max = {density_result.l_max}")
    print(f"  charge = {density_result.atoms.info['charge']}")
    print(f"  spin = {density_result.atoms.info['spin']}")
    print("Files:")
    print(f"  {output_dir / f'{stem}_polar_density_grid.npz'}")
    if not args.no_cube:
        print(f"  {output_dir / f'{stem}_charge_density.cube'}")
        print(f"  {output_dir / f'{stem}_spin_density.cube'}")
    print(f"  {output_dir / f'{stem}_charge_density_{args.plane}.png'}")
    print(f"  {output_dir / f'{stem}_spin_density_{args.plane}.png'}")


if __name__ == "__main__":
    main()
