"""Validator for PipeGen pipeline config JSON.

validate(config) -> list[Issue]

Rules are plain functions registered with @rule, so adding a new check is
just "write a function, decorate it" - no central dispatcher to edit.

Extending to catch AI-output regressions (outputs that aren't byte-identical
each run): don't diff raw JSON/text between runs. Instead run *this*
validator against every generated config as a structural/semantic contract
that must always pass, and separately snapshot-test the specific fields that
should be stable for a given prompt (e.g. pipeline type, num_classes,
target) while ignoring fields allowed to vary (e.g. model name choice,
comments, key order). Prefer comparing normalized/parsed structure over
exact strings, and consider property-based tests ("for any prompt
mentioning N classes, num_classes == N") instead of fixed golden files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

VALID_PRECISIONS = {"fp32", "fp16", "int8"}


@dataclass(frozen=True)
class Issue:
    rule: str
    severity: str  # "error" | "warning"
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message} ({self.rule})"


def _get(config: dict, path: str, default: Any = None) -> Any:
    """Dotted-path lookup that never raises, e.g. _get(cfg, 'model.precision')."""
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


Rule = Callable[[dict], "list[Issue]"]

_RULES: "list[Rule]" = [] #Empty Test Case List


def rule(func: Rule) -> Rule: #Appending/ Creating List of TC
    """Decorator that registers a validation rule with validate()."""
    _RULES.append(func)
    return func


@rule
def check_input_size_matches_resize(config: dict) -> "list[Issue]":
    input_shape = _get(config, "model.input_shape")
    resize = _get(config, "preprocess.resize")

    if input_shape is None or resize is None:
        return []

    if not isinstance(input_shape, list) or len(input_shape) < 2:
        return [Issue(
            rule="input_size_matches_resize",
            severity="error",
            field="model.input_shape",
            message=f"input_shape must have at least 2 dims (..., H, W), got {input_shape!r}",
        )]

    if not isinstance(resize, list) or len(resize) != 2:
        return [Issue(
            rule="input_size_matches_resize",
            severity="error",
            field="preprocess.resize",
            message=f"resize must be a [width, height] pair, got {resize!r}",
        )]

    spatial = list(input_shape[-2:])
    if spatial != list(resize):
        return [Issue(
            rule="input_size_matches_resize",
            severity="error",
            field="preprocess.resize",
            message=(
                f"model spatial input size {spatial} (from model.input_shape) "
                f"does not equal preprocess.resize {resize}"
            ),
        )]
    return []


@rule
def check_num_classes_matches_labels(config: dict) -> "list[Issue]":
    num_classes = _get(config, "num_classes")
    labels = _get(config, "labels")

    if num_classes is None or labels is None:
        return []

    if not isinstance(labels, list):
        return [Issue(
            rule="num_classes_matches_labels",
            severity="error",
            field="labels",
            message=f"labels must be a list, got {type(labels).__name__}",
        )]

    if num_classes != len(labels):
        return [Issue(
            rule="num_classes_matches_labels",
            severity="error",
            field="num_classes",
            message=f"num_classes ({num_classes}) does not equal len(labels) ({len(labels)})",
        )]
    return []


@rule
def check_precision_supported(config: dict) -> "list[Issue]":
    precision = _get(config, "model.precision")
    if precision is None:
        return []

    if precision not in VALID_PRECISIONS:
        return [Issue(
            rule="precision_supported",
            severity="error",
            field="model.precision",
            message=f"precision {precision!r} is not supported; expected one of {sorted(VALID_PRECISIONS)}",
        )]
    return []


@rule
def check_save_video_has_output_path(config: dict) -> "list[Issue]":
    save_video = _get(config, "output.save_video")
    output_path = _get(config, "output.output_path")

    if save_video is True and not output_path:
        return [Issue(
            rule="save_video_has_output_path",
            severity="error",
            field="output.output_path",
            message="output.save_video is true but output.output_path is not set",
        )]
    return []


def validate(config: dict) -> "list[Issue]":
    """Run all registered rules against a pipeline config and return found issues."""
    issues: "list[Issue]" = []
    for check in _RULES:
        issues.extend(check(config))
    return issues
