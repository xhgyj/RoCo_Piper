"""Enforce boundaries of migrated modules without claiming legacy compliance."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "rocobrick"
LAYERS = {
    "contracts": {"contracts"},
    "task_plan": {"contracts", "task_plan"},
    "scheduling": {"contracts", "task_plan", "scheduling"},
}


def imports(source: str, module: str) -> set[str]:
    """Return resolved static imports, including package-relative edges."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")[: -node.level]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Name)
                and function.id in {"__import__", "eval", "exec"}
            ) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            ):
                raise AssertionError("dynamic import/evaluation in offline core")
    return found


def validate_boundary(module: str, dependencies: set[str]) -> None:
    """Permit stdlib data processing and inward static dependencies only."""
    layer = module.split(".")[1]
    for dependency in dependencies:
        parts = dependency.split(".")
        if parts[0] == "rocobrick":
            assert len(parts) > 1 and parts[1] in LAYERS[layer], (
                f"{module} imports forbidden implementation {dependency}"
            )
        else:
            assert parts[0] in sys.stdlib_module_names, (
                f"{module} imports external framework {dependency}"
            )
            assert parts[0] not in {
                "subprocess",
                "socket",
                "ctypes",
                "multiprocessing",
            }, f"{module} imports side-effect transport {dependency}"


def test_new_core_imports_point_inward_and_have_no_cycles():
    """Do not import modules while auditing their framework independence."""
    edges: dict[str, set[str]] = {}
    for layer in LAYERS:
        files = sorted((ROOT / layer).rglob("*.py"))
        assert files, f"missing implemented layer {layer}"
        for path in files:
            module = "rocobrick." + ".".join(
                path.relative_to(ROOT).with_suffix("").parts
            )
            dependencies = imports(path.read_text(), module)
            validate_boundary(module, dependencies)
            edges[module] = dependencies
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(module: str) -> None:
        assert module not in visiting, f"import cycle at {module}"
        if module in visited:
            return
        visiting.add(module)
        for dependency in edges[module] & edges.keys():
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in edges:
        visit(module)


@pytest.mark.parametrize(
    "source",
    [
        "import rclpy",
        "from rocobrick.adapters.moveit import motion",
        "from rocobrick.skills import pick",
        "import subprocess",
        "from ..skills import pick",
    ],
)
def test_boundary_audit_detects_forbidden_dependencies(source: str):
    """Ensure the guard fails when a framework or outward edge is introduced."""
    with pytest.raises(AssertionError):
        validate_boundary(
            "rocobrick.scheduling.scheduler",
            imports(source, "rocobrick.scheduling.scheduler"),
        )


def test_boundary_audit_rejects_dynamic_import_escape():
    """Static architecture contracts cannot silently permit dynamic bypasses."""
    with pytest.raises(AssertionError, match="dynamic"):
        imports("__import__('rclpy')", "rocobrick.scheduling.scheduler")
