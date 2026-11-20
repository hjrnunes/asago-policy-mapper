"""Tests that the decision extra stays lazy: the default install never imports it.

Note: importing ``asago_policy_mapper.assessors.http_probability`` pulls in
``extract.screening`` through the package's lazy ``extract/__init__``, which
can import the pipeline and, transitively, ``openai`` (which bundles httpx).
That chain predates the assessor and is not this module's to change. What
these tests own is that the assessor module itself never imports httpx at
module level: the import sits inside the constructor, so the module's own
import graph adds nothing. The tests pin exactly that contract.
"""

import ast
import subprocess
import sys
from pathlib import Path

_MODULE = Path("src/asago_policy_mapper/assessors/http_probability.py")


def test_decision_extra_not_imported_by_default():
    """Importing the package and its assessors subpackage pulls no assessor module.

    A fresh interpreter that imports asago_policy_mapper and
    asago_policy_mapper.assessors must not import the HTTP assessor module,
    whose constructor is the only place httpx enters.
    """
    code = (
        "import sys\n"
        "import asago_policy_mapper\n"
        "import asago_policy_mapper.assessors\n"
        "assert 'asago_policy_mapper.assessors.http_probability' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_http_probability_module_has_no_module_level_httpx_import():
    """The assessor module's own import graph adds no httpx.

    Parse the module source and assert no module-level import of httpx:
    the assessor imports it inside __init__, so importing the module (for
    its types) never requires the decision extra.
    """
    tree = ast.parse(_MODULE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert "httpx" not in names, f"module-level import of httpx: {names}"
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "httpx":
            raise AssertionError(f"module-level import from httpx: {node.module}")


def test_decision_extra_declared_in_pyproject():
    """pyproject.toml declares the decision optional dependency with httpx."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text()

    assert 'decision = ["httpx' in text
