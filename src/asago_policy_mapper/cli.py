import json
import logging
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import typer
import yaml

from asago_policy_mapper.llm import LLMConfig, TokenTracker, create_client

app = typer.Typer()


@app.callback()
def main():
    """Asago Policy Mapper — policy risk extraction using AI Atlas Nexus."""
    # Attach a root handler so the pipeline's existing INFO stage logs are
    # visible. basicConfig is a no-op when logging is already configured, so
    # an embedding application (e.g. the battery runner) keeps its own setup.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


EXCLUDED_TAXONOMIES = {
    "mit-ai-risk-repository-causal",
    "ibm-granite-guardian",
    "nist-ai-rmf",
    "owasp-llm-2.0",
    "ailuminate-v1.0",
    "owasp-asi",
    "shieldgemma-taxonomy",
}


def _fail_validation(message: str) -> None:
    """Echo a validation error and exit with code 1."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _require_existing_policy_files(policy_files) -> None:
    """Fail validation unless every policy file exists."""
    for pf in policy_files:
        if not pf.exists():
            _fail_validation(f"{pf} does not exist")


def _explicit_option_names(param_names: tuple[str, ...]) -> frozenset[str]:
    """Collect the dashed CLI names of the parameters the user actually passed.

    Warnings fire only for explicitly passed options, so defaults (and
    values that arrive via environment variables, which click reports as
    ``ENVIRONMENT``) stay quiet. Called from inside the command body, where
    the current click context holds the parameter sources.
    """
    import click

    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return frozenset()
    # Boolean pairs (--expand-siblings/--no-expand-siblings) report one
    # parameter name, so the flag's actual polarity decides which dashed
    # name lands in the set; the store-true no_* flags map directly.
    boolean_pairs = {"expand_siblings", "query_gen"}
    explicit = set()
    for name in param_names:
        source = ctx.get_parameter_source(name)
        if source != click.core.ParameterSource.COMMANDLINE:
            continue
        dashed = "--" + name.replace("_", "-")
        if name in boolean_pairs and not ctx.params.get(name, False):
            dashed = "--no-" + name.replace("_", "-")
        explicit.add(dashed)
    return frozenset(explicit)


def _validate_extract_options(
    *,
    output_format,
    policy_files,
    base_url,
    model,
    retrieval,
    prefilter=False,
    explicit=frozenset(),
):
    """Validate extract options; exit on error. Returns the run's resolved plan.

    The plan decides whether the run needs an LLM, and the missing-endpoint
    error names the stages that call one for this exact combination of
    options (``plan.llm_stages``), so the message follows the flags.
    """
    from asago_policy_mapper.extract.plan import resolve_plan

    valid_formats = {"json", "yaml", "both"}
    if output_format not in valid_formats:
        _fail_validation(f"--output-format must be one of: {', '.join(sorted(valid_formats))}")

    _require_existing_policy_files(policy_files)

    if prefilter and retrieval.no_grounding:
        _fail_validation("--prefilter-profile requires grounding; it filters what grounding sees")

    if retrieval.rrf_min_score < 0:
        _fail_validation("--rrf-min-score must not be negative: it is a floor, never a ceiling")

    plan = resolve_plan(retrieval, prefilter=prefilter, explicit=explicit)
    if plan.needs_llm and (not base_url or not model):
        stages = ", ".join(plan.llm_stages)
        _fail_validation(f"--base-url/--model required: stages {stages} call an LLM")

    return plan


def _setup_llm(
    *,
    needs_llm,
    base_url,
    model,
    api_key,
    max_concurrent,
    max_tokens,
    output_token_parameter,
    service_tier,
    max_context,
    temperature,
    top_p,
    top_k,
):
    """Build the LLM config, token tracker and client for the run.

    ``needs_llm`` comes from the resolved run plan. LLM-free runs (pure
    IR evaluation) substitute "unused" endpoint values and skip client
    creation.
    """
    if not needs_llm:
        base_url = base_url or "unused"
        model = model or "unused"
    config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        output_token_parameter=output_token_parameter,
        service_tier=service_tier,
        max_context=max_context,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    tracker = TokenTracker()
    client = None if not needs_llm else create_client(config, tracker=tracker)
    return config, tracker, client


def _load_custom_risks(custom_taxonomy) -> list:
    """Load custom taxonomy YAML files, exiting on invalid input."""
    loaded_custom_risks: list = []
    if custom_taxonomy:
        from asago_policy_mapper.taxonomy import load_custom_taxonomy

        for tax_path in custom_taxonomy:
            try:
                custom_risks = load_custom_taxonomy(tax_path)
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from e
            loaded_custom_risks.extend(custom_risks)
            typer.echo(f"Loaded {len(custom_risks)} custom risks from {tax_path}")
    return loaded_custom_risks


def _load_risk_catalog(nexus_base_dir, loaded_custom_risks) -> tuple[list, int]:
    """Load Nexus risks (minus excluded taxonomies, with description overrides) and merge custom risks.

    Returns (all_risks, custom_risk_count); colliding custom risk ids are skipped with a warning.
    """
    from ai_atlas_nexus import AIAtlasNexus

    from asago_policy_mapper.taxonomy import apply_description_overrides

    nexus = AIAtlasNexus(base_dir=nexus_base_dir)
    all_risks = apply_description_overrides(
        r for r in nexus.get_all_risks() if getattr(r, "isDefinedByTaxonomy", "") not in EXCLUDED_TAXONOMIES
    )

    custom_risk_count = 0
    if loaded_custom_risks:
        seen_ids = {r.id for r in all_risks}
        for cr in loaded_custom_risks:
            if cr.id in seen_ids:
                typer.echo(f"Warning: custom risk '{cr.id}' collides with an existing risk - skipping", err=True)
            else:
                all_risks.append(cr)
                seen_ids.add(cr.id)
                custom_risk_count += 1
    return all_risks, custom_risk_count


def _warn_risk_text(all_risks, check_risk_text) -> None:
    """Echo the unterminated-description warning when --check-risk-text is set."""
    if check_risk_text:
        from asago_policy_mapper.taxonomy import warn_unterminated_descriptions

        flagged = warn_unterminated_descriptions(all_risks)
        typer.echo(
            f"Risk text check: {len(flagged)} of {len(all_risks)} risk descriptions lack closing punctuation"
            + (f": {', '.join(flagged)}" if flagged else ""),
            err=True,
        )


def _validate_prefilter_threshold(value, path) -> float:
    """Check the profile's threshold: a finite number in [0, 1].

    A bad threshold must fail at profile load, not mid-run: the rule only
    compares against it at decide time, so a string or NaN would
    otherwise surface as a ``TypeError`` deep inside the prefilter stage
    — or, for out-of-range values, filter incorrectly in silence.
    """
    if value is None:
        return 0.10
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail_validation(f"--prefilter-profile {path}: threshold must be a number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        _fail_validation(f"--prefilter-profile {path}: threshold must be within [0, 1], got {value!r}")
    return float(value)


def _load_prefilter_profile(path: Path) -> dict:
    """Read and parse the profile YAML; exit with a validation error on any failure."""
    if not path.is_file():
        _fail_validation(f"--prefilter-profile {path} does not exist or is not a file")
    try:
        profile = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError, UnicodeError) as e:
        _fail_validation(f"--prefilter-profile {path} could not be read as YAML: {e}")
    if not isinstance(profile, dict):
        _fail_validation(f"--prefilter-profile {path} must be a YAML mapping of endpoint fields plus threshold")
    return dict(profile)


def _validate_prefilter_numeric(
    profile: dict, key: str, path: Path, *, minimum: float | None = None, exclusive_minimum: bool = False
) -> None:
    """Check one optional numeric profile field fails fast when invalid.

    ``ProbabilityEndpoint`` is a plain dataclass: a wrong-typed
    ``timeout_s`` or ``max_batch`` from YAML would otherwise surface as
    a ``TypeError`` mid-run, inside the assessor. ``max_batch`` must be
    an integer (``range`` rejects float steps), and an explicit null is
    an error rather than a silently-omitted field. ``exclusive_minimum``
    turns the floor into a strict bound (``timeout_s`` must be positive,
    not zero: a zero timeout fails every request).
    """
    value = profile.get(key)
    if value is None:
        if key in profile:
            _fail_validation(f"--prefilter-profile {path}: {key} must not be null")
        return
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if key == "max_batch":
        if not isinstance(value, int) or isinstance(value, bool):
            _fail_validation(f"--prefilter-profile {path}: {key} must be an integer, got {value!r}")
    elif not numeric:
        _fail_validation(f"--prefilter-profile {path}: {key} must be a number, got {value!r}")
    if minimum is not None:
        below = value <= minimum if exclusive_minimum else value < minimum
        if below:
            relation = "greater than" if exclusive_minimum else "at least"
            _fail_validation(f"--prefilter-profile {path}: {key} must be {relation} {minimum}, got {value!r}")


def _validate_prefilter_strings(profile: dict, keys: tuple[str, ...], path: Path) -> None:
    """Check the endpoint's string fields are present as nonempty strings.

    A non-string ``base_url`` fails inside the assessor constructor, and
    a non-string ``api_key_env`` only fails mid-assessment; both belong
    at profile load.
    """
    for key in keys:
        value = profile.get(key)
        if not isinstance(value, str) or not value.strip():
            _fail_validation(f"--prefilter-profile {path}: {key} must be a nonempty string, got {value!r}")


def _validate_endpoint_base_url(base_url: str, label: str) -> None:
    """Check a profile URL is an http(s) URL with a host.

    ``urlsplit`` accepts scheme-less hosts, empty hosts and unknown
    schemes silently, and the assessor only surfaces the problem on the
    first request — a typo'd scheme would fail every request while the
    fail-open filter keeps everything, making a permanent
    misconfiguration indistinguishable from an outage. The check runs
    at profile load instead. ``label`` names the option in the error.
    """
    try:
        parts = urlsplit(base_url)
    except ValueError:
        _fail_validation(f"{label}: base_url is not a valid URL, got {base_url!r}")
    if parts.scheme not in ("http", "https") or not parts.hostname:
        _fail_validation(f"{label}: base_url must be an http(s) URL with a host, got {base_url!r}")


def _require_endpoint_api_key(api_key_env: str, base_url: str, label: str) -> None:
    """Fail when an offsite endpoint has no API key in its named variable.

    An offsite endpoint with an unset key would send policy text away
    without authentication, then fail every batch with an authorization
    error while the filter keeps everything: the run must refuse to
    start instead. Local endpoints need no key. The provider profile
    builders can adopt the same rule.
    """
    if os.environ.get(api_key_env):
        return
    from asago_policy_mapper.assessors.http_probability import _is_local_host

    if not _is_local_host(urlsplit(base_url).hostname):
        _fail_validation(
            f"{label}: endpoint {base_url!r} is offsite and {api_key_env} is unset; "
            "set the variable or point base_url at a local endpoint"
        )


def _validate_prefilter_provider(profile: dict, path: Path) -> str | None:
    """Check the optional provider key: None/generic, jev or laya.

    An unknown or non-string provider must fail at profile load: a typo
    would otherwise silently build the generic assessor against a provider
    profile. The valid set is the implemented mappings.
    """
    if "provider" not in profile:
        return None
    provider = profile.get("provider")
    if provider is None:
        return None
    if not isinstance(provider, str) or provider not in ("generic", "jev", "laya"):
        _fail_validation(
            f"--prefilter-profile {path}: provider must be one of 'generic', 'jev', 'laya', got {provider!r}"
        )
    return provider


def _build_prefilter(profile_path):
    """Load a prefilter profile YAML into the pipeline's ``GroundingPrefilter``.

    The profile carries the ``ProbabilityEndpoint`` fields plus
    ``threshold`` and an optional ``provider`` key selecting the mapping:
    absent/``generic`` builds the generic HTTP assessor, ``jev`` the Jev
    mapping (TypeSafe System One), ``laya`` the local in-process arm. The
    API key itself comes from the environment variable the profile names,
    never from the profile. ``None`` (no path) means no filter. Malformed
    or missing profiles exit with a validation error, as do installs
    without the extra the chosen provider needs.
    """
    if profile_path is None:
        return None

    path = Path(profile_path)
    profile = _load_prefilter_profile(path)
    provider = _validate_prefilter_provider(profile, path)
    profile.pop("provider", None)

    if provider in ("jev", "laya"):
        return _build_provider_prefilter(provider, profile, path)

    _validate_prefilter_strings(profile, ("base_url", "model", "api_key_env"), path)
    _validate_endpoint_base_url(profile["base_url"], f"--prefilter-profile {path}")
    _validate_prefilter_numeric(profile, "timeout_s", path, minimum=0.0, exclusive_minimum=True)
    _validate_prefilter_numeric(profile, "max_batch", path, minimum=1.0)
    threshold = _validate_prefilter_threshold(profile.pop("threshold", None), path)

    try:
        from asago_policy_mapper.assessors.http_probability import HttpProbabilityAssessor, ProbabilityEndpoint
        from asago_policy_mapper.extract.pipeline import GroundingPrefilter
        from asago_policy_mapper.extract.screening import DropThreshold
    except ImportError as e:
        _fail_validation(f"--prefilter-profile could not import its stack: {e}")
    _require_endpoint_api_key(profile["api_key_env"], profile["base_url"], f"--prefilter-profile {path}")

    try:
        endpoint = ProbabilityEndpoint(**profile)
    except TypeError as e:
        _fail_validation(f"--prefilter-profile {path} has invalid fields: {e}")
    try:
        assessor = HttpProbabilityAssessor(endpoint)
    except ImportError as e:
        _fail_validation(
            f"--prefilter-profile needs the decision extra (httpx); install with: uv sync --extra decision ({e})"
        )
    rule = DropThreshold(assessor, threshold)
    return GroundingPrefilter(assessor=assessor, rule=rule)


def _build_provider_prefilter(provider: str, profile: dict, path: Path):
    """Build a provider-mapped prefilter (jev or laya) from a profile.

    Provider endpoints default every field, so the profile may carry only
    the overrides and the threshold. The remaining keys validate with the
    same rules as the generic profile (minus the required base_url/model/
    api_key_env, which defaults fill), and unknown keys still fail loudly.
    """
    from asago_policy_mapper.extract.gates import ThresholdDrop
    from asago_policy_mapper.extract.pipeline import Prefilter

    _validate_prefilter_numeric(profile, "timeout_s", path, minimum=0.0)
    _validate_prefilter_numeric(profile, "max_batch", path, minimum=1.0)
    threshold = _validate_prefilter_threshold(profile.pop("threshold", None), path)
    for key in ("base_url", "model", "api_key_env"):
        if key in profile:
            _validate_prefilter_strings(profile, (key,), path)

    if provider == "jev":
        from asago_policy_mapper.assessors.jev import JevProbabilityAssessor, JevProbabilityEndpoint

        try:
            endpoint = JevProbabilityEndpoint(**profile)
        except TypeError as e:
            _fail_validation(f"--prefilter-profile {path} has invalid fields: {e}")
        try:
            assessor: object = JevProbabilityAssessor(endpoint)
        except ImportError as e:
            _fail_validation(
                f"--prefilter-profile needs the decision extra (httpx); install with: uv sync --extra decision ({e})"
            )
    else:
        from asago_policy_mapper.assessors.laya import LayaProbabilityAssessor, LayaProbabilityEndpoint

        laya_endpoint: LayaProbabilityEndpoint
        try:
            laya_endpoint = LayaProbabilityEndpoint(**profile)
        except TypeError as e:
            _fail_validation(f"--prefilter-profile {path} has invalid fields: {e}")
        try:
            assessor = LayaProbabilityAssessor(laya_endpoint)
        except ImportError as e:
            _fail_validation(f"--prefilter-profile provider 'laya' needs laya-mlx installed: {e}")
        except OSError as e:
            # Checkpoint-load failures (typo'd model, offline with no cache)
            # raise huggingface_hub errors that subclass OSError.
            _fail_validation(
                f"--prefilter-profile provider 'laya' could not load checkpoint {laya_endpoint.model!r}: {e}"
            )
    rule = ThresholdDrop(assessor, threshold)
    return Prefilter(assessor=assessor, rule=rule)


def _build_retrieval_config(
    *,
    bi_encoder_model,
    bi_encoder_api_key,
    bi_encoder_model_name,
    query_instruction,
    cross_encoder_model,
    cross_encoder_type,
    cross_encoder_model_name,
    colbert_model,
    chunk_max_tokens,
    top_n_accept,
    top_n_judge,
    min_score_floor,
    threshold_high,
    threshold_low,
    bm25_rescue_rank,
    rrf_min_score,
    no_cross_encoder,
    no_judge,
    no_grounding,
    judge_prompt,
    judge_context_tokens,
    expand_siblings,
    grounding_passes,
    grounding_batch_size,
    expansion_passes,
    no_causal_synthesis,
    query_gen,
):
    """Map the extract command's retrieval options onto RetrievalConfig."""
    from asago_policy_mapper.extract.models import RetrievalConfig

    return RetrievalConfig(
        bi_encoder_model=bi_encoder_model,
        bi_encoder_api_key=bi_encoder_api_key,
        bi_encoder_model_name=bi_encoder_model_name or None,
        query_instruction=query_instruction if query_instruction is not None else RetrievalConfig.query_instruction,
        cross_encoder_model=cross_encoder_model,
        cross_encoder_type=cross_encoder_type,
        cross_encoder_model_name=cross_encoder_model_name or None,
        colbert_model=colbert_model or None,
        chunk_max_tokens=chunk_max_tokens,
        top_n_accept=top_n_accept,
        top_n_judge=top_n_judge,
        min_score_floor=min_score_floor,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        bm25_rescue_rank=bm25_rescue_rank,
        rrf_min_score=rrf_min_score,
        use_cross_encoder=not no_cross_encoder,
        no_judge=no_judge,
        no_grounding=no_grounding,
        judge_prompt=judge_prompt,
        judge_context_tokens=judge_context_tokens,
        expand_siblings=expand_siblings,
        grounding_passes=grounding_passes,
        grounding_batch_size=grounding_batch_size,
        expansion_passes=expansion_passes,
        no_causal_synthesis=no_causal_synthesis,
        query_gen=query_gen,
    )


def _echo_extraction_banner(policy_files, all_risks, custom_risk_count) -> None:
    if custom_risk_count:
        typer.echo(
            f"Extracting risks from {len(policy_files)} document(s) "
            f"({len(all_risks) - custom_risk_count} Nexus + {custom_risk_count} custom risks loaded)..."
        )
    else:
        typer.echo(f"Extracting risks from {len(policy_files)} document(s) ({len(all_risks)} Nexus risks loaded)...")


def _run_pipeline(policy_files, client, config, all_risks, retrieval, ocr, run_extraction, prefilter=None, plan=None):
    """Run the extraction pipeline over the loaded risk catalog.

    The caller resolves run_extraction so its import happens at the same
    point as before the refactor, before the extraction banner echoes.
    A resolved plan is forwarded so the pipeline records the plan the CLI
    validated and warned about instead of re-resolving one without the
    explicit options (which would leave the metadata warnings empty).
    """
    kwargs = {} if plan is None else {"plan": plan}
    return run_extraction(
        documents=policy_files,
        client=client,
        config=config,
        risks=all_risks,
        retrieval=retrieval,
        ocr=ocr,
        prefilter=prefilter,
        **kwargs,
    )


def _attach_mitigations(result, nexus_base_dir) -> None:
    """Enrich matched risks with mitigations, threats and consequences from the Nexus data."""
    from asago_policy_mapper.extract.mitigations import (
        build_action_descriptions,
        build_risk_crossmap,
        enrich_with_mitigations,
        load_mitigation_index,
        load_risk_consequences,
        load_risk_threats,
    )

    mitigation_index = load_mitigation_index()
    if mitigation_index:
        action_descs = build_action_descriptions(nexus_base_dir)
        risk_crossmap = build_risk_crossmap(nexus_base_dir)
        risk_threats = load_risk_threats()
        risk_consequences = load_risk_consequences()
        enrich_with_mitigations(
            result.risks, mitigation_index, action_descs, risk_crossmap, risk_threats, risk_consequences
        )
        typer.echo(f"  Mitigations attached from {len(mitigation_index)} risk entries")


def _write_extraction_outputs(
    result,
    result_data,
    output,
    output_format,
    plan,
    tracker,
) -> None:
    """Write the extraction result as JSON/YAML and echo the run summary."""
    write_json = output_format in ("json", "both")
    write_yaml = output_format in ("yaml", "both")

    if write_json:
        json_path = output / "risk-extraction.json"
        json_path.write_text(json.dumps(result_data, indent=2))
        typer.echo(f"Risk extraction written to {json_path}")
    if write_yaml:
        yaml_path = output / "risk-extraction.yaml"
        yaml_path.write_text(yaml.dump(result_data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        typer.echo(f"Risk extraction written to {yaml_path}")
    typer.echo(f"  {len(result.risks)} risks matched")
    stats = result.retrieval_stats
    if not plan.grounding and plan.judge != "llm" and not plan.needs_llm:
        typer.echo(f"  {stats.auto_accepted} auto-accepted (IR-only, no LLM stages)")
    elif not plan.grounding and plan.judge != "llm":
        typer.echo(f"  {stats.auto_accepted} auto-accepted (IR-only, query generation)")
    elif not plan.grounding:
        typer.echo(f"  {stats.auto_accepted} auto-accepted, {stats.llm_judged} LLM-judged (no grounding)")
    elif plan.judge != "llm":
        typer.echo(
            f"  {stats.auto_accepted} auto-accepted, {stats.grounding_filtered} grounding-filtered (no cross-encoder)"
        )
    else:
        typer.echo(
            f"  {stats.auto_accepted} auto-accepted, {stats.llm_judged} LLM-judged,"
            f" {stats.grounding_filtered} grounding-filtered"
        )
    if plan.needs_llm:
        typer.echo(
            f"Token usage: {tracker.prompt_tokens:,} prompt + {tracker.completion_tokens:,} completion"
            f" = {tracker.total_tokens:,} total ({tracker.calls} calls)"
        )


def _write_extraction_reports(result_data, policy_files, output, ocr, chunk_max_tokens) -> None:
    """Write the HTML extraction report, then the document-centric annotation view."""
    from asago_policy_mapper.extract.report import build_risk_extraction_report

    report_path = build_risk_extraction_report(result_data, output / "risk-extraction.html")
    typer.echo(f"Report written to {report_path}")

    from asago_policy_mapper.extract.annotate import prepare_annotation_data
    from asago_policy_mapper.extract.report import build_annotation_report

    annotation_data = prepare_annotation_data(result_data, policy_files[0], ocr=ocr, chunk_max_tokens=chunk_max_tokens)
    ann_path = build_annotation_report(annotation_data, output / "risk-annotation.html")
    typer.echo(f"Annotation view written to {ann_path}")


@app.command()
def extract(
    policy_files: list[Path] = typer.Argument(..., help="Policy document(s) to extract risks from"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    base_url: str = typer.Option(None, "--base-url", envvar="POLICY_MAPPER_BASE_URL", help="LLM API base URL"),
    model: str = typer.Option(None, "--model", envvar="POLICY_MAPPER_MODEL", help="LLM model name"),
    api_key: str = typer.Option("none", "--api-key", envvar="POLICY_MAPPER_API_KEY", help="LLM API key"),
    nexus_base_dir: str = typer.Option(
        None,
        "--nexus-base-dir",
        envvar="NEXUS_BASE_DIR",
        help="Optional override: path to an ai-atlas-nexus clone (defaults to the installed package data)",
    ),
    custom_taxonomy: list[Path] = typer.Option(
        [], "--custom-taxonomy", "-t", help="Custom taxonomy YAML file(s) to load alongside Nexus risks"
    ),
    check_risk_text: bool = typer.Option(
        False,
        "--check-risk-text",
        help="Warn about loaded risks whose description lacks closing punctuation (possible truncation)",
    ),
    max_concurrent: int = typer.Option(32, "--max-concurrent", help="Max parallel LLM calls"),
    max_tokens: int = typer.Option(
        8192,
        "--max-tokens",
        envvar="POLICY_MAPPER_MAX_OUTPUT_TOKENS",
        help="Max LLM output tokens per call (default: 8192)",
    ),
    output_token_parameter: str = typer.Option(
        "max_tokens",
        "--output-token-parameter",
        envvar="POLICY_MAPPER_OUTPUT_TOKEN_PARAMETER",
        help="LLM output token parameter: max_tokens or max_completion_tokens (default: max_tokens)",
    ),
    service_tier: str | None = typer.Option(
        None,
        "--service-tier",
        envvar="POLICY_MAPPER_SERVICE_TIER",
        help="Optional service tier passed unchanged to the LLM endpoint (default: omitted)",
    ),
    max_context: int = typer.Option(
        0, "--max-context", help="Model context window in tokens (0=do not budget against context)"
    ),
    ocr: bool = typer.Option(False, "--ocr", help="Enable OCR for document conversion"),
    chunk_max_tokens: int = typer.Option(512, "--chunk-max-tokens", help="Max tokens per chunk (default: 512)"),
    top_n_accept: int = typer.Option(10, "--top-n-accept", help="Auto-accept top N candidates per chunk (rank-based)"),
    top_n_judge: int = typer.Option(10, "--top-n-judge", help="Send next N candidates to LLM judge (rank-based)"),
    min_score_floor: float = typer.Option(
        0.70, "--min-score-floor", help="Reject candidates below this score regardless of rank"
    ),
    threshold_high: float = typer.Option(
        None, "--threshold-high", help="Legacy: absolute auto-accept threshold (overrides rank-based)"
    ),
    threshold_low: float = typer.Option(
        None, "--threshold-low", help="Legacy: absolute discard threshold (overrides rank-based)"
    ),
    bi_encoder_model: str = typer.Option(
        "all-mpnet-base-v2",
        "--bi-encoder-model",
        envvar="POLICY_MAPPER_BI_ENCODER_MODEL",
        help="Bi-encoder model",
    ),
    bi_encoder_api_key: str = typer.Option(
        "none", "--bi-encoder-api-key", envvar="POLICY_MAPPER_BI_ENCODER_API_KEY", help="Bi-encoder API key"
    ),
    bi_encoder_model_name: str = typer.Option(
        None,
        "--bi-encoder-model-name",
        envvar="POLICY_MAPPER_BI_ENCODER_MODEL_NAME",
        help="Bi-encoder model name (overrides name derived from endpoint URL)",
    ),
    query_instruction: str = typer.Option(
        None,
        "--query-instruction",
        help="Instruction prefix for query encoding (default: built-in policy-risk instruction)",
    ),
    cross_encoder_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "--cross-encoder-model",
        envvar="POLICY_MAPPER_CROSS_ENCODER_MODEL",
        help="Cross-encoder model",
    ),
    cross_encoder_type: str = typer.Option(
        "score",
        "--cross-encoder-type",
        help=(
            "Cross-encoder API type: 'score' for /v1/score, 'rerank' for /v1/rerank, "
            "'generative' for /v1/completions logprob rerankers"
        ),
    ),
    cross_encoder_model_name: str = typer.Option(
        None,
        "--cross-encoder-model-name",
        envvar="POLICY_MAPPER_CROSS_ENCODER_MODEL_NAME",
        help="Cross-encoder model name (overrides name derived from endpoint URL)",
    ),
    bm25_rescue_rank: int = typer.Option(
        0, "--bm25-rescue-rank", help="BM25 rank cutoff for rescuing candidates past cross-encoder (0=disabled)"
    ),
    no_cross_encoder: bool = typer.Option(
        False, "--no-cross-encoder", help="Skip cross-encoder reranking and LLM judge; use RRF score floor instead"
    ),
    rrf_min_score: float = typer.Option(
        0.015,
        "--rrf-min-score",
        help="RRF floor; used with --no-cross-encoder and on every query-generation search (0 = no floor)",
    ),
    colbert_model: str = typer.Option(
        None,
        "--colbert-model",
        help="ColBERT model for late interaction retrieval (replaces bi-encoder + cross-encoder)",
    ),
    judge_prompt: str = typer.Option(
        "judge_risk",
        "--judge-prompt",
        help="Judge prompt template name (judge_risk, judge_risk_gepa, judge_risk_gepa_demos)",
    ),
    judge_context_tokens: int = typer.Option(
        0,
        "--judge-context-tokens",
        help="Token budget for judge-only contexts (0=use the grounding context's default padding)",
    ),
    expand_siblings: bool = typer.Option(
        True,
        "--expand-siblings/--no-expand-siblings",
        help="Expand to sibling risks after merge and ground against relevant chunks (default: enabled)",
    ),
    grounding_passes: int = typer.Option(
        3,
        "--grounding-passes",
        help="Number of per-chunk grounding passes; union of results reduces variance (default: 3)",
    ),
    grounding_batch_size: int = typer.Option(
        15,
        "--grounding-batch-size",
        help="Max candidate risks per grounding LLM call (default: 15; 0=no split)",
    ),
    expansion_passes: int = typer.Option(
        3,
        "--expansion-passes",
        help="Number of expansion grounding passes; union of results reduces variance (default: 3)",
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Skip LLM judge; auto-promote borderline candidates to accepted"
    ),
    no_grounding: bool = typer.Option(
        False, "--no-grounding", help="Skip LLM grounding; accepted candidates become matches without evidence"
    ),
    no_causal_synthesis: bool = typer.Option(
        False, "--no-causal-synthesis", help="Skip LLM causal chain synthesis; populate from static YAML only"
    ),
    query_gen: bool = typer.Option(
        True, "--query-gen/--no-query-gen", help="LLM query generation for retrieval (default: on)"
    ),
    prefilter_profile: Path | None = typer.Option(
        None,
        "--prefilter-profile",
        envvar="POLICY_MAPPER_PREFILTER_PROFILE",
        help="YAML profile for the opt-in pre-grounding filter (ProbabilityEndpoint fields plus threshold)",
    ),
    no_prefilter: bool = typer.Option(
        False,
        "--no-prefilter",
        help="Turn off the pre-grounding filter, overriding --prefilter-profile and POLICY_MAPPER_PREFILTER_PROFILE",
    ),
    output_format: str = typer.Option(
        "json", "--output-format", "-f", help="Output format: json, yaml, or both (default: json)"
    ),
    temperature: float = typer.Option(0.0, "--temperature", help="LLM sampling temperature (default: 0.0)"),
    top_p: float = typer.Option(None, "--top-p", help="LLM nucleus sampling top-p (omitted if not set)"),
    top_k: int = typer.Option(
        None, "--top-k", help="LLM top-k sampling (passed via extra_body for vLLM; omitted if not set)"
    ),
):
    """Extract risks from policy documents using hybrid retrieval."""
    # --no-prefilter wins over the environment variable: a CI run can
    # inherit POLICY_MAPPER_PREFILTER_PROFILE and still run unfiltered.
    if no_prefilter:
        prefilter_profile = None
    retrieval = _build_retrieval_config(
        bi_encoder_model=bi_encoder_model,
        bi_encoder_api_key=bi_encoder_api_key,
        bi_encoder_model_name=bi_encoder_model_name,
        query_instruction=query_instruction,
        cross_encoder_model=cross_encoder_model,
        cross_encoder_type=cross_encoder_type,
        cross_encoder_model_name=cross_encoder_model_name,
        colbert_model=colbert_model,
        chunk_max_tokens=chunk_max_tokens,
        top_n_accept=top_n_accept,
        top_n_judge=top_n_judge,
        min_score_floor=min_score_floor,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        bm25_rescue_rank=bm25_rescue_rank,
        rrf_min_score=rrf_min_score,
        no_cross_encoder=no_cross_encoder,
        no_judge=no_judge,
        no_grounding=no_grounding,
        judge_prompt=judge_prompt,
        judge_context_tokens=judge_context_tokens,
        expand_siblings=expand_siblings,
        grounding_passes=grounding_passes,
        grounding_batch_size=grounding_batch_size,
        expansion_passes=expansion_passes,
        no_causal_synthesis=no_causal_synthesis,
        query_gen=query_gen,
    )
    explicit = _explicit_option_names(
        (
            "no_judge",
            "no_grounding",
            "no_cross_encoder",
            "no_causal_synthesis",
            "expand_siblings",
            "query_gen",
            "rrf_min_score",
            "judge_context_tokens",
            "prefilter_profile",
        )
    )

    plan = _validate_extract_options(
        output_format=output_format,
        policy_files=policy_files,
        base_url=base_url,
        model=model,
        retrieval=retrieval,
        prefilter=prefilter_profile is not None,
        explicit=explicit,
    )

    for warning in plan.warnings:
        typer.echo(f"Warning: {warning}", err=True)

    config, tracker, client = _setup_llm(
        needs_llm=plan.needs_llm,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        output_token_parameter=output_token_parameter,
        service_tier=service_tier,
        max_context=max_context,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Build and validate the profile before creating any output: a bad
    # profile fails the run without leaving an empty output directory.
    prefilter = _build_prefilter(prefilter_profile)

    output.mkdir(parents=True, exist_ok=True)

    loaded_custom_risks = _load_custom_risks(custom_taxonomy)
    # Normalize once for every consumer (catalog + mitigation loaders): a blank
    # --nexus-base-dir / NEXUS_BASE_DIR behaves exactly like unset.
    nexus_base_dir_override = nexus_base_dir or None
    all_risks, custom_risk_count = _load_risk_catalog(nexus_base_dir_override, loaded_custom_risks)
    _warn_risk_text(all_risks, check_risk_text)

    from asago_policy_mapper.extract.pipeline import run_extraction

    _echo_extraction_banner(policy_files, all_risks, custom_risk_count)

    result = _run_pipeline(
        policy_files, client, config, all_risks, retrieval, ocr, run_extraction, prefilter, plan=plan
    )

    result.token_usage = tracker.to_dict()

    _attach_mitigations(result, nexus_base_dir_override)

    result_data = result.model_dump()

    _write_extraction_outputs(
        result,
        result_data,
        output,
        output_format,
        plan,
        tracker,
    )

    _write_extraction_reports(result_data, policy_files, output, ocr, chunk_max_tokens)


@app.command(name="annotate")
def annotate_cmd(
    run_dir: Path = typer.Argument(..., help="Directory containing risk-extraction.json"),
    source_document: Path = typer.Argument(..., help="Original policy document (PDF, DOCX, etc.)"),
    ground_truth: Path = typer.Option(None, "--ground-truth", "-g", help="Ground truth YAML for comparison mode"),
    ocr: bool = typer.Option(False, "--ocr", help="Enable OCR for document conversion"),
    chunk_max_tokens: int = typer.Option(512, "--chunk-max-tokens", help="Max tokens per chunk"),
):
    """Generate a document-centric annotation view from a completed extraction run."""
    json_path = run_dir / "risk-extraction.json"
    yaml_path = run_dir / "risk-extraction.yaml"
    if json_path.exists():
        extracted_path = json_path
    elif yaml_path.exists():
        extracted_path = yaml_path
    else:
        typer.echo(f"Error: no risk-extraction.json or .yaml found in {run_dir}", err=True)
        raise typer.Exit(1)

    extraction_data = (
        json.loads(extracted_path.read_text())
        if extracted_path.suffix == ".json"
        else yaml.safe_load(extracted_path.read_text())
    )

    from asago_policy_mapper.extract.annotate import prepare_annotation_data
    from asago_policy_mapper.extract.report import build_annotation_report

    annotation_data = prepare_annotation_data(
        extraction_data,
        source_document,
        ground_truth_path=ground_truth,
        extracted_path=extracted_path,
        ocr=ocr,
        chunk_max_tokens=chunk_max_tokens,
    )
    ann_path = build_annotation_report(annotation_data, run_dir / "risk-annotation.html")

    summary = annotation_data["summary"]
    typer.echo(f"Annotation view: {summary['total_risks']} risks, {summary['total_evidence_spans']} evidence spans")
    typer.echo(f"  Matched: {summary['matched_spans']}, Unmatched: {summary['unmatched_spans']}")
    if annotation_data.get("eval"):
        ev = annotation_data["eval"]
        typer.echo(
            f"  GT: {ev['matched']} matched, {len(ev['spurious'])} spurious, {len(ev['missing'])} missing"
            f" — P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f}"
        )
        revised_dir = run_dir / "revised_gt"
        revised_dir.mkdir(exist_ok=True)
        typer.echo(f"  Save revised GT to: {revised_dir}/")
    typer.echo(f"  Written to {ann_path}")


@app.command(name="eval")
def eval_cmd(
    run_dir: Path = typer.Argument(..., help="Directory containing risk-extraction.json"),
    ground_truth: Path = typer.Option(
        None, "--ground-truth", "-g", help="Ground truth YAML file (default: evals/ground_truth/{name}.yaml)"
    ),
    min_recall: float = typer.Option(0.80, "--min-recall", help="Minimum recall threshold"),
    min_precision: float = typer.Option(0.60, "--min-precision", help="Minimum precision threshold"),
):
    """Evaluate a risk extraction run against ground truth."""
    json_path = run_dir / "risk-extraction.json"
    yaml_path = run_dir / "risk-extraction.yaml"
    if json_path.exists():
        extracted_path = json_path
    elif yaml_path.exists():
        extracted_path = yaml_path
    else:
        typer.echo(f"Error: no risk-extraction.json or .yaml found in {run_dir}", err=True)
        raise typer.Exit(1)

    if ground_truth is None:
        evals_dir = Path(__file__).parent.parent.parent / "evals" / "ground_truth"
        ground_truth = evals_dir / f"{run_dir.name}.yaml"

    if not ground_truth.exists():
        typer.echo(f"Error: ground truth not found at {ground_truth}", err=True)
        raise typer.Exit(1)

    from asago_policy_mapper.evals.eval import evaluate_extraction

    result = evaluate_extraction(
        ground_truth,
        extracted_path,
        policy_name=run_dir.name,
        min_recall=min_recall,
        min_precision=min_precision,
    )

    eval_path = run_dir / "eval.json"
    eval_path.write_text(json.dumps(result, indent=2))

    raw = extracted_path.read_text()
    if extracted_path.suffix == ".yaml":
        extraction_data = yaml.safe_load(raw)
    else:
        extraction_data = json.loads(raw)
    extraction_data["eval"] = result

    if json_path.exists():
        json_path.write_text(json.dumps(extraction_data, indent=2))
    if yaml_path.exists():
        yaml_path.write_text(yaml.dump(extraction_data, default_flow_style=False, sort_keys=False, allow_unicode=True))

    from asago_policy_mapper.extract.report import build_risk_extraction_report

    report_path = build_risk_extraction_report(extraction_data, run_dir / "risk-extraction.html")

    status = "PASS" if result["pass"] else "FAIL"
    typer.echo(f"Eval: {result['policy']} — {status}")
    typer.echo(f"  Precision: {result['precision']:.3f} (threshold: {min_precision})")
    typer.echo(f"  Recall:    {result['recall']:.3f} (threshold: {min_recall})")
    typer.echo(f"  F1:        {result['f1']:.3f}")
    typer.echo(
        f"  Matched:   {result['matched']}/{result['total_expected']} expected, {result['total_extracted']} extracted"
    )
    if result["missing"]:
        typer.echo(f"  Missing:   {', '.join(result['missing'])}")
    if result["spurious"]:
        typer.echo(f"  Spurious:  {', '.join(result['spurious'])}")
    typer.echo(f"  Written to {eval_path}")
    typer.echo(f"  Report updated at {report_path}")
