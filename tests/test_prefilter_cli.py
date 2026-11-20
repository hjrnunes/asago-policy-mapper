"""CLI tests for the prefilter profile.

``--prefilter-profile PATH`` loads a YAML file with the
``ProbabilityAnswerEndpoint`` fields plus ``threshold`` and turns it into the
pipeline's ``GroundingPrefilter``. The API key comes from the environment
variable the profile names, never from the profile itself. The option's
own environment variable is ``POLICY_MAPPER_PREFILTER_PROFILE``.
Validation fails when the profile is combined with ``--no-grounding``.
"""

import re
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from asago_policy_mapper.cli import app

runner = CliRunner()


def _plain_help(result) -> str:
    """Return the help text free of ANSI codes and wrap artifacts.

    Rich (which typer uses for help) interleaves escape codes with the
    option names when it colors its output — on GitHub Actions it
    always does — and it truncates long option names when the table is
    narrow. The assertions we care about are about the option set, so
    ``_help_result`` pins the rendering (a terminal type rich accepts
    and a wide table) and this helper strips the escape codes and
    collapses the layout whitespace, so a name can match even where the
    renderer wrapped it.
    """
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result.stdout)
    return re.sub(r"\s+", " ", text)


def _help_result():
    """Invoke ``extract --help`` with a deterministic rendering.

    The rendered width must not depend on the machine running the
    tests: a dumb ``TERM`` makes rich hardcode an 80-column table (too
    narrow for ``--prefilter-profile``, truncating it), and a forced
    terminal without a real tty also lands at 80. Pinning ``TERM`` to a
    non-dumb value and ``COLUMNS`` to a wide table keeps every option
    name intact regardless of the ambient environment.
    """
    return runner.invoke(app, ["extract", "--help"], env={"TERM": "xterm", "COLUMNS": "200"})


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


def _validate_kwargs(tmp_path, retrieval, prefilter=False, **overrides):
    kwargs = dict(
        output_format="json",
        policy_files=[_policy(tmp_path)],
        base_url="http://localhost:8000/v1",
        model="test-model",
        retrieval=retrieval,
        prefilter=prefilter,
        explicit=frozenset(),
    )
    kwargs.update(overrides)
    return kwargs


def test_validate_rejects_prefilter_with_no_grounding(tmp_path, profile_path):
    """--prefilter-profile with --no-grounding fails validation."""
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    kwargs = _validate_kwargs(
        tmp_path,
        retrieval=RetrievalConfig(no_grounding=True, query_gen=False),
        prefilter=True,
    )
    with pytest.raises(typer.Exit) as excinfo:
        _validate_extract_options(**kwargs)
    assert excinfo.value.exit_code == 1


def test_validate_allows_prefilter_with_grounding(tmp_path, profile_path):
    """--prefilter-profile passes validation on a grounded run."""
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    assert (
        _validate_extract_options(
            **_validate_kwargs(tmp_path, retrieval=RetrievalConfig(query_gen=False), prefilter=True)
        ).needs_llm
        is True
    )


def test_validate_allows_unset_prefilter_with_no_grounding(tmp_path):
    """No profile + --no-grounding still validates (the filter is opt-in)."""
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    # needs_llm either way; must not exit.
    _validate_extract_options(
        **_validate_kwargs(tmp_path, retrieval=RetrievalConfig(no_grounding=True, query_gen=False))
    )


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
    result = _help_result()
    assert result.exit_code == 0
    assert "--prefilter-profile" in _plain_help(result)


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
    """A profile without a threshold gets DropThreshold's 0.10 default."""
    from asago_policy_mapper.cli import _build_prefilter

    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump({"base_url": "http://localhost:8080", "model": "m", "api_key_env": "K"}))
    prefilter = _build_prefilter(path)
    assert prefilter is not None
    assert prefilter.rule.threshold == 0.10


def test_build_prefilter_rejects_missing_endpoint_fields(tmp_path):
    """A profile missing required ProbabilityAnswerEndpoint fields fails clearly."""
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
    """The extract command builds the GroundingPrefilter and hands it to run_extraction."""
    from unittest.mock import patch

    from asago_policy_mapper.extract import pipeline

    captured = {}

    def fake_run_extraction(**kwargs):
        captured.update(kwargs)
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
    # The pipeline call forwards the resolved client and the ocr flag.
    assert captured["client"] is not None
    assert captured["ocr"] is False


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


# --- validation error messages ---


def _validation_error(profile_kwargs, tmp_path, monkeypatch=None):
    """Run _build_prefilter on a bad profile and return the message printed to stderr."""
    import asago_policy_mapper.cli as cli

    profile = dict(PROFILE)
    profile.update(profile_kwargs)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(profile))

    recorded = {}
    original_echo = typer.echo

    def fake_echo(message=None, *args, **kwargs):
        recorded.setdefault("message", message)
        original_echo(message, *args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(typer, "echo", fake_echo):
        try:
            cli._build_prefilter(path)
        except typer.Exit:
            pass
        else:
            raise AssertionError("expected a validation exit")
    return str(recorded.get("message", ""))


def test_prefilter_validation_messages_name_the_field_and_path(tmp_path):
    """Each validation error names the profile path and the offending field."""
    cases = [
        ({"threshold": "x"}, "threshold must be a number"),
        ({"threshold": 1.5}, "threshold must be within [0, 1]"),
        ({"timeout_s": "x"}, "timeout_s must be a number"),
        ({"timeout_s": -1.0}, "timeout_s must be greater than 0.0"),
        ({"timeout_s": 0}, "timeout_s must be greater than 0.0"),
        ({"max_batch": "x"}, "max_batch must be an integer"),
        ({"max_batch": 16.0}, "max_batch must be an integer"),
        ({"max_batch": None}, "max_batch must not be null"),
        ({"max_batch": 0}, "max_batch must be at least 1.0"),
        ({"base_url": 123}, "base_url must be a nonempty string"),
        ({"model": 123}, "model must be a nonempty string"),
        ({"api_key_env": 123}, "api_key_env must be a nonempty string"),
    ]
    for overrides, fragment in cases:
        message = _validation_error(overrides, tmp_path)
        assert fragment in message, f"{fragment!r} missing from {message!r}"
        assert "bad.yaml" in message, f"path missing from {message!r}"


def test_prefilter_threshold_boundaries_are_inclusive(tmp_path):
    """0.0 and 1.0 are valid thresholds; values outside fail."""
    from asago_policy_mapper.cli import _build_prefilter

    for edge in (0.0, 1.0):
        profile = dict(PROFILE)
        profile["threshold"] = edge
        path = tmp_path / "edge.yaml"
        path.write_text(yaml.safe_dump(profile))
        prefilter = _build_prefilter(path)
        assert prefilter is not None
        assert prefilter.rule.threshold == edge


def test_prefilter_minimum_boundaries(tmp_path):
    """max_batch 1 is valid; timeout_s must be strictly positive.

    A zero timeout fails every request, so 0 is rejected, not floored.
    """
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["max_batch"] = 1
    path = tmp_path / "floor.yaml"
    path.write_text(yaml.safe_dump(profile))
    prefilter = _build_prefilter(path)
    assert prefilter is not None
    assert prefilter.assessor.endpoint.max_batch == 1

    profile = dict(PROFILE)
    profile["timeout_s"] = 0
    path = tmp_path / "zero-timeout.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)


def test_missing_profile_and_bad_yaml_name_the_path(tmp_path, capsys):
    """The load errors name the path and the reason."""
    from asago_policy_mapper.cli import _build_prefilter

    with pytest.raises(typer.Exit):
        _build_prefilter(tmp_path / "nope.yaml")
    assert "nope.yaml does not exist or is not a file" in capsys.readouterr().err

    bad = tmp_path / "bad.yaml"
    bad.write_text("{unclosed")
    with pytest.raises(typer.Exit):
        _build_prefilter(bad)
    assert "could not be read as YAML" in capsys.readouterr().err

    listing = tmp_path / "list.yaml"
    listing.write_text("- a\n")
    with pytest.raises(typer.Exit):
        _build_prefilter(listing)
    assert "must be a YAML mapping" in capsys.readouterr().err


def test_no_grounding_refusal_message_names_the_option(tmp_path, capsys):
    """The refusal names --prefilter-profile, grounding, and the reason."""
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    kwargs = _validate_kwargs(
        tmp_path,
        retrieval=RetrievalConfig(no_grounding=True, query_gen=False),
        prefilter=True,
    )
    with pytest.raises(typer.Exit):
        _validate_extract_options(**kwargs)
    err = capsys.readouterr().err
    assert err.strip() == "Error: --prefilter-profile requires grounding; it filters what grounding sees"


def test_needs_llm_requires_base_url_or_model_each(tmp_path, capsys):
    """A run needing LLM calls with only one of base_url/model fails.

    The guard is ``or``: either missing setting alone must fail, not just
    both together.
    """
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    for base_url, model in ((None, "m"), ("http://x/v1", None)):
        kwargs = _validate_kwargs(
            tmp_path,
            retrieval=RetrievalConfig(query_gen=False),
            base_url=base_url,
            model=model,
        )
        with pytest.raises(typer.Exit):
            _validate_extract_options(**kwargs)
        err = capsys.readouterr().err
        assert err.strip() == (
            "Error: --base-url/--model required: stages judge, grounding, variant_grounding, expansion, "
            "causal_synthesis call an LLM"
        )


def test_output_format_error_lists_the_formats(tmp_path, capsys):
    """An invalid --output-format names the valid formats, comma-separated."""
    from asago_policy_mapper.cli import _validate_extract_options
    from asago_policy_mapper.extract.models import RetrievalConfig

    kwargs = _validate_kwargs(
        tmp_path,
        retrieval=RetrievalConfig(query_gen=False),
        output_format="xml",
    )
    with pytest.raises(typer.Exit):
        _validate_extract_options(**kwargs)
    err = capsys.readouterr().err
    assert "both, json, yaml" in err


def test_import_failure_messages_name_the_missing_stack(tmp_path, profile_path, monkeypatch, capsys):
    """Each import failure prints its specific advice, not a bare error."""
    import builtins
    import unittest.mock

    import asago_policy_mapper.cli as cli

    # 1. The stack import fails: the message says the stack could not import.
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "asago_policy_mapper.assessors.http_probability":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    with unittest.mock.patch("builtins.__import__", side_effect=failing_import):
        with pytest.raises(typer.Exit):
            cli._build_prefilter(profile_path)
    assert "could not import its stack" in capsys.readouterr().err

    # 2. Endpoint construction with bad fields names the invalid fields.
    profile = dict(PROFILE)
    profile["bogus_key"] = 1
    bad = tmp_path / "fields.yaml"
    bad.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        cli._build_prefilter(bad)
    assert "invalid fields" in capsys.readouterr().err

    # 3. The httpx import inside the assessor fails: the decision-extra advice.
    import asago_policy_mapper.assessors.http_probability as hpa

    class FailingAssessor:
        def __init__(self, endpoint):
            raise ImportError("httpx not installed")

    with unittest.mock.patch.object(hpa, "HttpProbabilityAssessor", FailingAssessor):
        with pytest.raises(typer.Exit):
            cli._build_prefilter(profile_path)
    err = capsys.readouterr().err
    assert "needs the decision extra (httpx)" in err
    assert "uv sync --extra decision" in err


# --- base_url shape ---


def test_build_prefilter_rejects_schemeless_base_url(tmp_path, capsys):
    """A base_url without a scheme fails at load, not on the first request."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["base_url"] = "localhost:8080"
    path = tmp_path / "schemeless.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)
    assert "must be an http(s) URL with a host" in capsys.readouterr().err


def test_build_prefilter_rejects_hostless_base_url(tmp_path, capsys):
    """A base_url with a scheme but no host fails at load."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["base_url"] = "http://"
    path = tmp_path / "hostless.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)
    assert "must be an http(s) URL with a host" in capsys.readouterr().err


def test_build_prefilter_rejects_non_http_scheme(tmp_path, capsys, monkeypatch):
    """A typo'd or non-http scheme fails at load, not by failing every request.

    With fail-open, a scheme typo would otherwise disable the filter
    silently for the whole run: every request errors, the rule bypasses,
    and the run completes unfiltered.
    """
    from asago_policy_mapper.cli import _build_prefilter

    monkeypatch.setenv("PROB_API_KEY", "test-key")  # offsite key present; the scheme is what fails
    for bad in ("ftp://probability.example.com/v1", "htp://probability.example.com/v1"):
        profile = dict(PROFILE)
        profile["base_url"] = bad
        path = tmp_path / "scheme.yaml"
        path.write_text(yaml.safe_dump(profile))
        with pytest.raises(typer.Exit):
            _build_prefilter(path)
        assert "must be an http(s) URL with a host" in capsys.readouterr().err


def test_build_prefilter_rejects_malformed_base_url(tmp_path, capsys):
    """A URL urlsplit cannot parse fails as a validation error, not a traceback."""
    from asago_policy_mapper.cli import _build_prefilter

    profile = dict(PROFILE)
    profile["base_url"] = "http://[::1"  # unterminated IPv6 literal
    path = tmp_path / "malformed.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        _build_prefilter(path)
    err = capsys.readouterr().err
    assert "not a valid URL" in err
    assert "Invalid IPv6" not in err  # the raw parser error, not a traceback


# --- API key presence ---


def test_build_prefilter_fails_offsite_without_key(monkeypatch, profile_path, capsys):
    """An offsite endpoint with an unset key variable refuses to start."""
    import asago_policy_mapper.cli as cli

    monkeypatch.delenv("PROB_API_KEY", raising=False)
    profile = dict(PROFILE)
    profile["base_url"] = "https://probability.example.com/v1"
    path = profile_path.parent / "offsite.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(typer.Exit):
        cli._build_prefilter(path)
    err = capsys.readouterr().err
    assert "offsite" in err
    assert "PROB_API_KEY is unset" in err


def test_build_prefilter_allows_offsite_with_key(monkeypatch, profile_path):
    """An offsite endpoint passes when the named variable is set."""
    from asago_policy_mapper.cli import _build_prefilter

    monkeypatch.setenv("PROB_API_KEY", "test-key")
    profile = dict(PROFILE)
    profile["base_url"] = "https://probability.example.com/v1"
    path = profile_path.parent / "offsite.yaml"
    path.write_text(yaml.safe_dump(profile))
    prefilter = _build_prefilter(path)
    assert prefilter is not None
    assert prefilter.assessor.endpoint.base_url == "https://probability.example.com/v1"


def test_build_prefilter_allows_local_without_key(monkeypatch, profile_path):
    """A local endpoint needs no key: replay and development setups stay keyless."""
    from asago_policy_mapper.cli import _build_prefilter

    monkeypatch.delenv("PROB_API_KEY", raising=False)
    prefilter = _build_prefilter(profile_path)  # PROFILE points at localhost
    assert prefilter is not None
    assert prefilter.assessor.endpoint.base_url == "http://localhost:8080"


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
    result = _help_result()
    assert result.exit_code == 0
    assert "--no-prefilter" in _plain_help(result)


# --- a bad profile must not create the output directory ---


def test_bad_profile_creates_no_output_dir(tmp_path, profile_path, monkeypatch):
    """A profile that fails validation leaves no empty output directory behind.

    The profile loads before ``output.mkdir``: an invalid profile exits
    before any filesystem change.
    """
    profile = dict(PROFILE)
    profile["base_url"] = "https://probability.example.com/v1"
    offsite = tmp_path / "offsite.yaml"
    offsite.write_text(yaml.safe_dump(profile))
    monkeypatch.delenv("PROB_API_KEY", raising=False)

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
            str(offsite),
        ],
    )
    assert result.exit_code == 1
    assert not out.exists()
