#!/usr/bin/env python3
"""Extract initial MACE per-species features h^(0) and elemental E0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from ase.data import chemical_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a MACE .model file and extract the per-species initial "
            "embedding h^(0) together with the elemental baseline energy E0."
        )
    )
    parser.add_argument("model", type=Path, help="Path to a MACE .model file")
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Restrict output to a single element symbol, e.g. C",
    )
    parser.add_argument(
        "--atomic-number",
        type=int,
        default=None,
        help="Restrict output to a single atomic number, e.g. 6",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    return parser.parse_args()


def symbol_from_z(z: int) -> str:
    if 0 <= z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z={z}"


def as_python_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().cpu().to(torch.float64).view(-1).tolist()


def build_species_records(model: torch.nn.Module) -> Dict[str, Any]:
    atomic_numbers = [int(z) for z in model.atomic_numbers.detach().cpu().tolist()]
    dtype = next(model.parameters()).dtype
    one_hot = torch.eye(len(atomic_numbers), dtype=dtype)

    with torch.no_grad():
        h0 = model.node_embedding(one_hot)
        e0 = model.atomic_energies_fn(one_hot)

    records = []
    for index, z in enumerate(atomic_numbers):
        records.append(
            {
                "index_in_model_table": index,
                "atomic_number": z,
                "symbol": symbol_from_z(z),
                "h0": as_python_list(h0[index]),
                "E0": as_python_list(e0[index]),
            }
        )

    node_embedding_irreps = str(model.node_embedding.linear.irreps_out)
    extra_embeddings = None
    if hasattr(model, "embedding_specs"):
        extra_embeddings = model.embedding_specs

    return {
        "model_path": None,
        "num_elements": len(atomic_numbers),
        "atomic_numbers": atomic_numbers,
        "node_embedding_irreps": node_embedding_irreps,
        "interpretation": (
            "For standard MACE, h^(0) is obtained by linearly embedding the "
            "species one-hot vector. Only 0e channels are present at t=0. "
            "Geometry enters later through edge radial functions and spherical "
            "harmonics in the first interaction block."
        ),
        "extra_embeddings": extra_embeddings,
        "species": records,
    }


def filter_species(data: Dict[str, Any], symbol: str | None, z: int | None) -> Dict[str, Any]:
    if symbol is None and z is None:
        return data

    symbol_upper = symbol.capitalize() if symbol is not None else None
    filtered = []
    for record in data["species"]:
        if symbol_upper is not None and record["symbol"] != symbol_upper:
            continue
        if z is not None and record["atomic_number"] != z:
            continue
        filtered.append(record)

    data = dict(data)
    data["species"] = filtered
    return data


def print_text(data: Dict[str, Any]) -> None:
    print(f"model_path: {data['model_path']}")
    print(f"num_elements: {data['num_elements']}")
    print(f"atomic_numbers: {data['atomic_numbers']}")
    print(f"node_embedding_irreps: {data['node_embedding_irreps']}")
    print("interpretation:")
    print(f"  {data['interpretation']}")

    if data["extra_embeddings"] is not None:
        print("extra_embeddings:")
        print(json.dumps(data["extra_embeddings"], indent=2, sort_keys=True))

    if not data["species"]:
        print("No species matched the requested filter.")
        return

    for record in data["species"]:
        print()
        print(
            f"[{record['index_in_model_table']}] "
            f"{record['symbol']} (Z={record['atomic_number']})"
        )
        print(f"E0: {record['E0']}")
        print(f"h^(0): {record['h0']}")


def main() -> None:
    args = parse_args()
    model = torch.load(args.model, map_location="cpu")
    data = build_species_records(model)
    data["model_path"] = str(args.model.resolve())
    data = filter_species(data, args.symbol, args.atomic_number)

    if args.json:
        print(json.dumps(data, indent=2))
        return
    print_text(data)


if __name__ == "__main__":
    main()
