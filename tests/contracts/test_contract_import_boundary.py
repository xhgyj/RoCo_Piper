"""The contracts package stays a leaf: stdlib and itself only."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import rocobrick.contracts

_BANNED_ROOTS = frozenset(
    {
        "rclpy",
        "moveit",
        "numpy",
        "scipy",
        "yaml",
        "pydantic",
        "typing_extensions",
    }
)


def _package_dir() -> Path:
    """Return the on-disk directory of the contracts package.

    Returns:
        The directory containing the contract modules.
    """
    return Path(rocobrick.contracts.__file__).parent


def test_contract_modules_import_only_stdlib_and_their_package() -> None:
    """No contract module may import business code, ROS, MoveIt or frameworks."""
    modules = sorted(_package_dir().glob("*.py"))
    assert modules
    violations: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                assert node.module is not None
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in _BANNED_ROOTS:
                    violations.append(f"{path.name}: {name}")
                elif root == "rocobrick" and not (
                    name == "rocobrick.contracts"
                    or name.startswith("rocobrick.contracts.")
                ):
                    violations.append(f"{path.name}: {name}")
                elif root != "rocobrick" and root not in (sys.stdlib_module_names):
                    violations.append(f"{path.name}: {name}")
    assert violations == []


def test_importing_contracts_does_not_import_business_modules() -> None:
    """A fresh interpreter importing contracts pulls no platform modules."""
    source = _package_dir().parents[1]
    script = (
        "import sys, rocobrick.contracts;"
        "print(','.join(sorted("
        "name for name in sys.modules if name.startswith('rocobrick.')"
        ")))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(source)},
    )
    imported = set(completed.stdout.strip().split(","))
    assert imported <= {
        "rocobrick",
        "rocobrick.contracts",
        "rocobrick.contracts._validation",
        "rocobrick.contracts.evidence",
        "rocobrick.contracts.failures",
        "rocobrick.contracts.identity",
        "rocobrick.contracts.motion",
        "rocobrick.contracts.phases",
    }
