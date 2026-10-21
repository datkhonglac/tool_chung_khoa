"""Kiểm tra config/settings.toml khi khởi động."""

import copy
import tomllib

import pytest

from vnsa.config import SETTINGS_FILE, ConfigError, check, validate


@pytest.fixture
def cfg():
    with SETTINGS_FILE.open("rb") as f:
        return tomllib.load(f)


def test_current_settings_valid(cfg):
    assert validate(cfg) == []


def test_weights_not_forced_to_100(cfg):
    # code tự chia lại trọng số: tổng khác 100 (định chế tài chính đang là 110) vẫn hợp lệ
    c = copy.deepcopy(cfg)
    c["fa_score"]["weights"]["sinh_loi"] = 40
    assert validate(c) == []


def test_negative_weight(cfg):
    c = copy.deepcopy(cfg)
    c["ta_score"]["weights"]["xu_huong"] = -5
    assert any("[ta_score.weights] xu_huong" in e for e in validate(c))


def test_swapped_threshold(cfg):
    c = copy.deepcopy(cfg)
    c["fa_score"]["thresholds"]["roe"] = [0.25, 0.05]            # đảo mức xấu / tốt
    c["fa_score"]["thresholds"]["debt_to_equity"] = [0.3, 1.5]   # số nhỏ là tốt mà đảo
    errors = validate(c)
    assert any("roe" in e and "NHỎ" in e for e in errors)
    assert any("debt_to_equity" in e and "LỚN" in e for e in errors)


def test_percent_typed_as_whole_number(cfg):
    c = copy.deepcopy(cfg)
    c["valuation"]["risk_free"] = 3.5
    assert any("risk_free" in e for e in validate(c))


def test_growth_above_cost_of_equity(cfg):
    c = copy.deepcopy(cfg)
    c["valuation"]["terminal_growth"] = 0.2
    assert any("terminal_growth" in e for e in validate(c))


def test_rating_thresholds_must_decrease(cfg):
    c = copy.deepcopy(cfg)
    c["recommend"]["rating_thresholds"] = [0.05, 0.15, -0.05, -0.15]
    assert any("rating_thresholds" in e for e in validate(c))


def test_missing_section_and_message(cfg):
    c = copy.deepcopy(cfg)
    del c["plan"]
    with pytest.raises(ConfigError, match=r"thiếu mục \[plan\]"):
        check(c)
