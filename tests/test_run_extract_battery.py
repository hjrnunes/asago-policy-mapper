"""Tests for the battery runner's extract-command construction.

The battery runner shells out to the CLI, so the flags it passes are its
contract: never a redundant or warning-triggering one.
"""

import sys
from pathlib import Path

import pytest

# The mutmut sandbox and the sandbox-isolation canary copy tests/ without
# the repository root, where run_extract_battery.py lives. Fall back to the
# enclosing checkout's root so ordinary mutmut runs keep collecting this
# module, which covers the battery's plan-based endpoint decision. Skip
# only when the script exists nowhere, as in the canary's scratch copy.
# Appending (not inserting) keeps any path mutmut itself puts ahead, such
# as mutants/src, in charge of imports.
for _root in (
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
):
    if (_root / "run_extract_battery.py").is_file():
        if str(_root) not in sys.path:
            sys.path.append(str(_root))
        break
else:
    pytest.skip("run_extract_battery.py is not present in this checkout", allow_module_level=True)

from run_extract_battery import _build_extract_command  # noqa: E402


def _command(**overrides) -> list[str]:
    kwargs = dict(
        policies=[Path("p.md")],
        out=Path("out/run1"),
        base_url="http://localhost:8000/v1",
        model="test-model",
    )
    kwargs.update(overrides)
    return _build_extract_command(**kwargs)


def test_expand_siblings_not_passed_when_it_matches_the_cli_default():
    """The positive --expand-siblings form never appears on the command line.

    Expansion defaults to on in the CLI, so passing the positive form adds
    nothing — and on --no-grounding runs it would print an ignored-option
    warning. Only a declined expansion passes a flag.
    """
    assert "--expand-siblings" not in _command()
    assert "--expand-siblings" not in _command(no_grounding=True)

    cmd = _command(expand_siblings=False)
    assert "--no-expand-siblings" in cmd


def test_ir_battery_command_carries_no_endpoint_dependence():
    """IR battery runs pass exactly the flags that switch each stage off."""
    cmd = _command(no_judge=True, no_grounding=True, query_gen=False, no_cross_encoder=True)
    assert "--no-judge" in cmd
    assert "--no-grounding" in cmd
    assert "--no-query-gen" in cmd
    assert "--no-cross-encoder" in cmd
