"""CLI tests for the prefilter profile (asago-uzfb1.7 step 4).

``--prefilter-profile PATH`` loads a YAML file with the
``ProbabilityEndpoint`` fields plus ``threshold`` and turns it into the
pipeline's ``Prefilter``. The API key comes from the environment
variable the profile names, never from the profile itself. The option's
own environment variable is ``POLICY_MAPPER_PREFILTER_PROFILE``.
Validation fails when the profile is combined with ``--no-grounding``.
"""

from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from asago_policy_mapper.cli import app

runner = CliRunner()

PROFILE = {
    "base_url": "http://localhost:8080",
    "model": "prob-model",
    "api_key_env": "PROB_API_KEY",
    "timeout_s": 5.0,
    "max_batch": 16,
    "threshold": 0.10,
}


@pytest.fixture
def profile_path(tmp_path):
    path = tmp_path / "prefilter.yaml"
    path.write_text(yaml.safe_dump(PROFILE))
    return path


def _policy(tmp_path):
    policy = tmp_path / "policy.txt"
    policy.write_text("test policy")
    return policy


# --- _validate_extract_options ---


def test_validate_rejects_prefilter_with_no_grounding(tmp_path, profile_path):
    """--prefilter-profile with --no-grounding fails validation."""
    from asago_policy_mapper.cli import _validate_extract_options

    kwargs = dict(
        output_format="json",
        policy_files=[_policy(tmp_path)],
        base_url="http://localhost:8000/v1",
        model="test-model",
        no_judge=False,
        no_grounding=True,
        expand_siblings=True,
        query_gen=False,
        prefilter_profile=profile_path,
    )
    with pytest.raises(typer.Exit) as excinfo:
        _validate_extract_options(**kwargs)
    assert excinfo.value.exit_code == 1


def test_validate_allows_prefilter_with_grounding(tmp_path, profile_path):
    """--prefilter-profile passes validation on a grounded run."""
    from asago_policy_mapper.cli import _validate_extract_options

    kwargs = dict(
        output_format="json",
        policy_files=[_policy(tmp_path)],
        base_url="http://localhost:8000/v1",
        model="test-model",
        no_judge=False,
        no_grounding=False,
        expand_siblings=True,
        query_gen=False,
        prefilter_profile=profile_path,
    )
    assert _validate_extract_options(**kwargs) is True


def test_validate_allows_unset_prefilter_with_no_grounding(tmp_path):
    """No profile + --no-grounding still validates (the filter is opt-in)."""
    from asago_policy_mapper.cli import _validate_extract_options

    kwargs = dict(
        output_format="json",
        policy_files=[_policy(tmp_path)],
        base_url="http://localhost:8000/v1",
        model="test-model",
        no_judge=False,
        no_grounding=True,
        expand_siblings=True,
        query_gen=False,
        prefilter_profile=None,
    )
    assert _validate_extract_options(**kwargs) is False or True  # needs_llm either way; must not exit


def test_extract_rejects_prefilter_profile_with_no_grounding_cli(tmp_path, profile_path):
    """The CLI command fails with exit code 1 on the bad combination."""
    result = runner.invoke(
        app,
        [
            "extract",
            str(_policy(tmp_path)),
            "-o",
            str(tmp_path / "out"),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--no-grounding",
            "--prefilter-profile",
            str(profile_path),
        ],
    )
    assert result.exit_code == 1
    assert "--prefilter-profile" in result.output or "no-grounding" in result.output


def test_extract_help_lists_prefilter_profile():
    """The option appears in the extract command's help."""
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    # Typer's rich help renderer truncates long option names at the default
    # terminal width.
    assert "--prefilter-prof" in result.stdout


def test_extract_prefilter_profile_envvar_attached():
    """The option's environment variable is POLICY_MAPPER_PREFILTER_PROFILE."""
    from typer.main import get_command

    command = get_command(app).commands["extract"]
    params = {p.name: p for p in command.params}
    assert "prefilter_profile" in params
    registered = params["prefilter_profile"].envvar
    if isinstance(registered, str):
        registered = [registered]
    assert "POLICY_MAPPER_PREFILTER_PROFILE" in (registered or [])


# --- _build_prefilter ---


def test_build_prefilter_reads_profile(tmp_path, profile_path):
    """A profile builds an HttpProbabilityAssessor under DropThreshold."""
    from asago_policy_mapper.cli import _build_prefilter

    prefilter = _build_prefilter(profile_path)
    assert prefilter is not None
    assert prefilter.assessor.endpoint.base_url == "http://localhost:8080"
    assert prefilter.assessor.endpoint.model == "prob-model"
    assert prefilter.assessor.endpoint.api_key_env == "PROB_API_KEY"
    assert prefilter.assessor.endpoint.timeout_s == 5.0
    assert prefilter.assessor.endpoint.max_batch == 16
    assert prefilter.rule.threshold == 0.10
    assert prefilter.rule.rule_id == "drop-threshold/1"


def test_build_prefilter_none_without_profile():
    """No path means no filter: the option is opt-in."""
    from asago_policy_mapper.cli import _build_prefilter

    assert _build_prefilter(None) is None


def test_build_prefilter_threshold_defaults(tmp_path):
    """A profile without a threshold gets DropThreshold's own default.

    The CLI reads the rule's ``DEFAULT_THRESHOLD`` constant rather than
    repeating the value, so the default the profile validation accepts
    and the default the rule applies cannot diverge; this test fails if
    either side reintroduces a private copy of the number.
    """
    from asago_policy_mapper.cli import _build_prefilter
    from asago_policy_mapper.extract.screening import DEFAULT_THRESHOLD

    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump({"base_url": "http://localhost:8080", "model": "m", "api_key_env": "K"}))
    prefilter = _build_prefilter(path)
    assert prefilter is not None
    assert prefilter.rule.threshold == DEFAULT_THRESHOLD


def test_build_prefilter_rejects_missing_endpoint_fields(tmp_path):
    """A profile missing required ProbabilityEndpoint fields fails clearly."""
    from asago_policy_mapper.cli import _build_prefilter

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"base_url": "http://localhost:8080"}))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_missing_file_fails(tmp_path):
    """A profile path that does not exist fails validation."""
    from asago_policy_mapper.cli import _build_prefilter

    with pytest.raises(typer.Exit):
        _build_prefilter(tmp_path / "nope.yaml")


def test_build_prefilter_rejects_unknown_keys(tmp_path):
    """Extra YAML keys fail loudly: a typo must not silently drop a setting."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["future_field"] = "ignored"
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_rejects_non_mapping(tmp_path):
    """A profile that is not a YAML mapping fails clearly."""
    from asago_policy_mapper.cli import _build_prefilter

    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


# --- wiring: the profile reaches run_extraction ---


def test_extract_passes_prefilter_to_pipeline(tmp_path, profile_path, monkeypatch):
    """The extract command builds the Prefilter and hands it to run_extraction."""
    from unittest.mock import patch

    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["prefilter"] = kwargs.get("prefilter")
        return pipeline._empty_result(kwargs["documents"])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)

    with (
        patch("ai_atlas_nexus.AIAtlasNexus", FakeNexus),
        patch("asago_policy_mapper.extract.report.build_risk_extraction_report") as fake_report,
        patch("asago_policy_mapper.extract.annotate.prepare_annotation_data") as fake_annotate,
        patch("asago_policy_mapper.extract.report.build_annotation_report") as fake_ann_report,
    ):
        fake_report.return_value = tmp_path / "r.html"
        fake_annotate.return_value = {}
        fake_ann_report.return_value = tmp_path / "a.html"
        result = runner.invoke(
            app,
            [
                "extract",
                str(_policy(tmp_path)),
                "-o",
                str(tmp_path / "out"),
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
                "--prefilter-profile",
                str(profile_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["prefilter"] is not None
    assert captured["prefilter"].rule.threshold == 0.10


def test_extract_envvar_loads_profile(tmp_path, profile_path, monkeypatch):
    """POLICY_MAPPER_PREFILTER_PROFILE carries the path when the flag is absent."""
    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["prefilter"] = kwargs.get("prefilter")
        return pipeline._empty_result(kwargs["documents"])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)
    monkeypatch.setenv("POLICY_MAPPER_PREFILTER_PROFILE", str(profile_path))

    from unittest.mock import patch

    with (
        patch("ai_atlas_nexus.AIAtlasNexus", FakeNexus),
        patch("asago_policy_mapper.extract.report.build_risk_extraction_report") as fake_report,
        patch("asago_policy_mapper.extract.annotate.prepare_annotation_data") as fake_annotate,
        patch("asago_policy_mapper.extract.report.build_annotation_report") as fake_ann_report,
    ):
        fake_report.return_value = tmp_path / "r.html"
        fake_annotate.return_value = {}
        fake_ann_report.return_value = tmp_path / "a.html"
        result = runner.invoke(
            app,
            [
                "extract",
                str(_policy(tmp_path)),
                "-o",
                str(tmp_path / "out"),
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["prefilter"] is not None
    assert captured["prefilter"].assessor.endpoint.model == "prob-model"


def test_build_prefilter_rejects_non_numeric_threshold(tmp_path):
    """A quoted threshold fails at profile load, not mid-run."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["threshold"] = "0.10"
    path = tmp_path / "str.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_rejects_out_of_range_threshold(tmp_path):
    """A threshold outside [0, 1] fails at profile load."""
    from asago_policy_mapper.cli import _build_prefilter

    for bad in (-0.1, 1.5):
        profile = dict(PROFILE)
        profile["threshold"] = bad
        path = tmp_path / "range.yaml"
        path.write_text(yaml.safe_dump(profile))
        with pytest.raises(typer.Exit):
            _build_prefilter(path)


def test_build_prefilter_rejects_non_numeric_endpoint_fields(tmp_path):
    """A wrong-typed timeout_s or max_batch fails at profile load."""
    from asago_policy_mapper.cli import _build_prefilter

    for key in ("timeout_s", "max_batch"):
        profile = dict(PROFILE)
        profile[key] = "abc"
        path = tmp_path / "field.yaml"
        path.write_text(yaml.safe_dump(profile))
        with pytest.raises(typer.Exit):
            _build_prefilter(path)


def test_build_prefilter_rejects_float_max_batch(tmp_path):
    """max_batch must be an integer: range() rejects float steps at batching time."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["max_batch"] = 16.0
    path = tmp_path / "float-batch.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_rejects_null_max_batch(tmp_path):
    """An explicit null max_batch is an error, not a silently-omitted field."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["max_batch"] = None
    path = tmp_path / "null-batch.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_rejects_non_string_endpoint_fields(tmp_path):
    """base_url, model and api_key_env must be nonempty strings at load."""
    from asago_policy_mapper.cli import _build_prefilter

    for key in ("base_url", "model", "api_key_env"):
        profile = dict(PROFILE)
        profile[key] = 123
        path = tmp_path / "strfield.yaml"
        path.write_text(yaml.safe_dump(profile))
        with pytest.raises(typer.Exit):
            _build_prefilter(path)


def test_build_prefilter_rejects_bad_encoding(tmp_path):
    """A profile that is not valid text fails as a validation error, not a traceback."""
    from asago_policy_mapper.cli import _build_prefilter

    path = tmp_path / "binary.yaml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_build_prefilter_rejects_directory(tmp_path):
    """A directory path fails as a profile validation error, not a traceback."""
    from asago_policy_mapper.cli import _build_prefilter

    with pytest.raises(typer.Exit):
        _build_prefilter(tmp_path)


def test_profile_carries_no_credentials(tmp_path, profile_path):
    """The profile's serialized form never contains a key, only the env name."""
    from asago_policy_mapper.cli import _build_prefilter

    prefilter = _build_prefilter(profile_path)
    assert prefilter is not None
    endpoint_dict = prefilter.assessor.endpoint.to_dict()
    assert endpoint_dict["api_key_env"] == "PROB_API_KEY"
    assert "api_key" not in endpoint_dict
    assert Path(profile_path).read_text().count("PROB_API_KEY") == 1


# --- --no-prefilter override ---


def test_no_prefilter_flag_overrides_envvar_profile(tmp_path, profile_path, monkeypatch):
    """--no-prefilter wins over a profile set through the environment variable."""
    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["prefilter"] = kwargs.get("prefilter")
        return pipeline._empty_result(kwargs["documents"])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)
    monkeypatch.setenv("POLICY_MAPPER_PREFILTER_PROFILE", str(profile_path))

    from unittest.mock import patch

    with (
        patch("ai_atlas_nexus.AIAtlasNexus", FakeNexus),
        patch("asago_policy_mapper.extract.report.build_risk_extraction_report") as fake_report,
        patch("asago_policy_mapper.extract.annotate.prepare_annotation_data") as fake_annotate,
        patch("asago_policy_mapper.extract.report.build_annotation_report") as fake_ann_report,
    ):
        fake_report.return_value = tmp_path / "r.html"
        fake_annotate.return_value = {}
        fake_ann_report.return_value = tmp_path / "a.html"
        result = runner.invoke(
            app,
            [
                "extract",
                str(_policy(tmp_path)),
                "-o",
                str(tmp_path / "out"),
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
                "--no-prefilter",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["prefilter"] is None


def test_no_prefilter_flag_beats_explicit_profile(tmp_path, profile_path, monkeypatch):
    """--no-prefilter turns the filter off even with --prefilter-profile given.

    The off switch is unconditional: a run that asks for no filter runs
    unfiltered, whatever else set a profile.
    """
    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured["prefilter"] = kwargs.get("prefilter")
        return pipeline._empty_result(kwargs["documents"])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)

    class _Risk:
        id = "r1"
        name = "r1"
        description = "A risk."
        isDefinedByTaxonomy = "test"
        isPartOf = ""

    class FakeNexus:
        def __init__(self, base_dir=None):
            pass

        def get_all_risks(self):
            return [_Risk()]

    import asago_policy_mapper.cli as cli

    monkeypatch.setattr(cli, "_attach_mitigations", lambda result, nexus_base_dir: None)

    from unittest.mock import patch

    with (
        patch("ai_atlas_nexus.AIAtlasNexus", FakeNexus),
        patch("asago_policy_mapper.extract.report.build_risk_extraction_report") as fake_report,
        patch("asago_policy_mapper.extract.annotate.prepare_annotation_data") as fake_annotate,
        patch("asago_policy_mapper.extract.report.build_annotation_report") as fake_ann_report,
    ):
        fake_report.return_value = tmp_path / "r.html"
        fake_annotate.return_value = {}
        fake_ann_report.return_value = tmp_path / "a.html"
        result = runner.invoke(
            app,
            [
                "extract",
                str(_policy(tmp_path)),
                "-o",
                str(tmp_path / "out"),
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
                "--prefilter-profile",
                str(profile_path),
                "--no-prefilter",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["prefilter"] is None


def test_no_prefilter_help_lists_the_flag():
    """The override appears in the extract command's help."""
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--no-prefilter" in result.stdout


# --- a bad profile must not create the output directory ---


def test_bad_profile_creates_no_output_dir(tmp_path, profile_path, monkeypatch):
    """A profile that fails validation leaves no empty output directory behind.

    The profile loads before ``output.mkdir``: an invalid profile exits
    before any filesystem change.
    """
    profile = dict(PROFILE)
    profile["threshold"] = 1.5
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(profile))

    out = tmp_path / "never-created"
    result = runner.invoke(
        app,
        [
            "extract",
            str(_policy(tmp_path)),
            "-o",
            str(out),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--prefilter-profile",
            str(bad),
        ],
    )
    assert result.exit_code == 1
    assert not out.exists()
