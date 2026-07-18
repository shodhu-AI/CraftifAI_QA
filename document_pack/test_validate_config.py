import copy
import json
from pathlib import Path

import pytest

from validate_config import validate

CONFIG_PATH = Path(__file__).parent / "config.json"


@pytest.fixture
def buggy_config():
    """The repo's own config.json, loaded as-is.

    It trips all four required rules at once:
      - model.input_shape spatial size [640, 640] != preprocess.resize [512, 512]
      - num_classes (80) != len(labels) (2: "people", "car")
      - output.save_video is true with no output.output_path
    (precision "fp16" here is valid, so that rule alone is exercised separately below.)
    """
    with CONFIG_PATH.open() as f:
        return json.load(f)


@pytest.fixture
def corrected_config(buggy_config):
    """A hand-fixed version of config.json that should pass every rule."""
    cfg = copy.deepcopy(buggy_config)
    cfg["preprocess"]["resize"] = [640, 640]
    cfg["labels"] = [f"class_{i}" for i in range(80)]
    cfg["num_classes"] = 80
    cfg["model"]["precision"] = "fp16"
    cfg["output"]["output_path"] = "/tmp/out.mp4"
    return cfg


def test_buggy_config_reports_all_expected_issues(buggy_config):
    issues = validate(buggy_config)
    rules_hit = {issue.rule for issue in issues}

    assert "input_size_matches_resize" in rules_hit
    assert "num_classes_matches_labels" in rules_hit
    assert "save_video_has_output_path" in rules_hit
    # precision "fp16" is valid, so that rule should NOT fire here
    assert "precision_supported" not in rules_hit


def test_corrected_config_has_no_issues(corrected_config):
    assert validate(corrected_config) == []


def test_input_size_mismatch_is_flagged(corrected_config):
    corrected_config["preprocess"]["resize"] = [512, 512]
    issues = validate(corrected_config)
    assert any(i.rule == "input_size_matches_resize" for i in issues)


def test_input_size_match_is_not_flagged(corrected_config):
    corrected_config["model"]["input_shape"] = [1, 3, 512, 512]
    corrected_config["preprocess"]["resize"] = [512, 512]
    issues = validate(corrected_config)
    assert not any(i.rule == "input_size_matches_resize" for i in issues)


def test_num_classes_mismatch_is_flagged(corrected_config):
    corrected_config["num_classes"] = 3
    issues = validate(corrected_config)
    assert any(i.rule == "num_classes_matches_labels" for i in issues)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_valid_precisions_are_not_flagged(corrected_config, precision):
    corrected_config["model"]["precision"] = precision
    issues = validate(corrected_config)
    assert not any(i.rule == "precision_supported" for i in issues)


@pytest.mark.parametrize("precision", ["int4", "fp64", "INT8", "", None])
def test_invalid_precisions_are_flagged(corrected_config, precision):
    corrected_config["model"]["precision"] = precision
    issues = validate(corrected_config)
    if precision is None:
        # Missing precision is a separate concern from an unsupported one.
        assert not any(i.rule == "precision_supported" for i in issues)
    else:
        assert any(i.rule == "precision_supported" for i in issues)


def test_save_video_without_output_path_is_flagged(corrected_config):
    corrected_config["output"]["save_video"] = True
    corrected_config["output"].pop("output_path", None)
    issues = validate(corrected_config)
    assert any(i.rule == "save_video_has_output_path" for i in issues)


def test_save_video_false_does_not_require_output_path(corrected_config):
    corrected_config["output"]["save_video"] = False
    corrected_config["output"].pop("output_path", None)
    issues = validate(corrected_config)
    assert not any(i.rule == "save_video_has_output_path" for i in issues)


def test_save_video_with_empty_output_path_is_flagged(corrected_config):
    corrected_config["output"]["save_video"] = True
    corrected_config["output"]["output_path"] = ""
    issues = validate(corrected_config)
    assert any(i.rule == "save_video_has_output_path" for i in issues)


def test_missing_sections_do_not_crash():
    # A minimal/partial config should be handled gracefully, not raise.
    assert validate({}) == []

    issues = validate({"model": {"precision": "int4"}})
    assert [i.rule for i in issues] == ["precision_supported"]


def test_issue_str_is_human_readable(corrected_config):
    corrected_config["num_classes"] = 3
    issues = validate(corrected_config)
    assert issues, "expected at least one issue"
    text = str(issues[0])
    assert issues[0].severity.upper() in text
    assert issues[0].field in text
