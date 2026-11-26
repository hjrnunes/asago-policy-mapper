"""Cross-process determinism of sibling expansion (issue #150 part 1).

expand_with_siblings attributes a shared sibling — a risk reachable from
several found risks — to the first found risk that reaches it. Iterating
the found-risk set directly made that attribution depend on PYTHONHASHSEED,
so two processes running the same pipeline with the same merged output
attributed the sibling differently and built different grounding groups.
The subprocess runs here pin identical attribution across two hash seeds
that iterate the found-risk set in opposite orders.
"""

import json
import subprocess
import sys

from asago_policy_mapper.extract.expand import expand_with_siblings

_PROBE = """
import json

from asago_policy_mapper.extract.expand import expand_with_siblings

expansion_graph = {
    "R-001": {"R-SHARED"},
    "R-002": {"R-SHARED"},
}
risk_lookup = {
    "R-SHARED": {"name": "Shared risk", "description": "Desc shared"},
}
result = expand_with_siblings({"R-001", "R-002"}, expansion_graph, risk_lookup)
print(
    json.dumps(
        [
            {
                "risk_id": er.risk_id,
                "source_risk_id": er.source_risk_id,
                "risk_name": er.risk_name,
            }
            for er in result
        ]
    )
)
"""

# These two seeds iterate the two-element found-risk set in opposite orders
# (seed 5: R-001 first; seed 2: R-002 first), so they produce different
# attributions on any order that follows set iteration. If the mutmut
# sandbox or a different Python build changes which seeds flip the order,
# re-verify with: PYTHONHASHSEED=<seed> python -c "print(list({'R-001', 'R-002'}))"
_HASH_SEEDS = (5, 2)


def test_shared_sibling_attribution_is_seed_independent():
    """The shared sibling is attributed identically under both hash seeds."""
    outputs = []
    for seed in _HASH_SEEDS:
        run = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
        )
        assert run.returncode == 0, run.stderr
        outputs.append(json.loads(run.stdout))

    assert outputs[0] == outputs[1]


def test_shared_sibling_attributed_to_lowest_found_risk():
    """The sorted iteration attributes the sibling to the lowest found risk.

    R-001 sorts before R-002, so the shared sibling is attributed to R-001
    regardless of the order the merged risk ids arrived in.
    """
    result = expand_with_siblings(
        {"R-002", "R-001"},
        {
            "R-001": {"R-SHARED"},
            "R-002": {"R-SHARED"},
        },
        {
            "R-SHARED": {"name": "Shared risk", "description": "Desc shared"},
        },
    )

    assert [(er.risk_id, er.source_risk_id) for er in result] == [("R-SHARED", "R-001")]
