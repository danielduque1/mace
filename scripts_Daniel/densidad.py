#!/usr/bin/env python3
"""
Minimal script to extract the final density coefficients from PolarMACE.

The mapping to the paper notation is:

    p_up    = spin_charge_density[:, 0, :]
    p_down  = spin_charge_density[:, 1, :]
    p_total = p_up + p_down
    p_spin  = p_up - p_down
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import mace_polar


def prepare_atoms(
    atoms: Atoms,
    charge: float | None = None,
    spin: float | None = None,
) -> Atoms:
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


def extract_density_coefficients(
    atoms: Atoms,
    model: str,
    *,
    device: str = "cpu",
    default_dtype: str = "float32",
    charge: float | None = None,
    spin: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Run PolarMACE and return the final density coefficients."""

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

    spin_charge_density = np.asarray(
        calculator.results["spin_charge_density"], dtype=np.float64
    )
    density_coefficients = np.asarray(
        calculator.results["density_coefficients"], dtype=np.float64
    )

    p_up = spin_charge_density[:, 0, :].copy()
    p_down = spin_charge_density[:, 1, :].copy()
    p_total = p_up + p_down
    p_spin = p_up - p_down

    return {
        "atomic_numbers": np.asarray(atoms.numbers, dtype=np.int64),
        "positions": atoms.get_positions(),
        "p_up": p_up,
        "p_down": p_down,
        "p_total": p_total,
        "p_spin": p_spin,
        "density_coefficients": density_coefficients,
        "spin_charge_density": spin_charge_density,
        "l_max": l_max,
        "sigma": sigma,
        "charge": float(atoms.info["charge"]),
        "spin": float(atoms.info["spin"]),
    }


def extract_density_coefficients_from_file(
    input_path: str | Path,
    model: str,
    *,
    index: str = "0",
    device: str = "cpu",
    default_dtype: str = "float32",
    charge: float | None = None,
    spin: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    atoms = read(input_path, index=index)
    return extract_density_coefficients(
        atoms=atoms,
        model=model,
        device=device,
        default_dtype=default_dtype,
        charge=charge,
        spin=spin,
    )


def save_coefficients_npz(
    output_path: str | Path,
    data: dict[str, np.ndarray | float | int],
) -> None:
    np.savez_compressed(output_path, **data)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--charge", type=float, default=None)
    parser.add_argument("--spin", type=float, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .npz output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = extract_density_coefficients_from_file(
        input_path=args.input,
        model=args.model,
        index=args.index,
        device=args.device,
        default_dtype=args.default_dtype,
        charge=args.charge,
        spin=args.spin,
    )

    print("PolarMACE density coefficients")
    print(f"  atoms: {data['p_total'].shape[0]}")
    print(f"  coefficients per atom: {data['p_total'].shape[1]}")
    print(f"  l_max: {data['l_max']}")
    print(f"  sigma: {data['sigma']:.6f} A")
    print(f"  charge: {data['charge']}")
    print(f"  spin: {data['spin']}")

    if args.output is not None:
        save_coefficients_npz(args.output, data)
        print(f"  saved: {args.output}")


if __name__ == "__main__":
    main()
