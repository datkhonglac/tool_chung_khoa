from pathlib import Path

import pandas as pd
import pytest

from vnsa.data.cleaning import validate_prices
from vnsa.data.providers.csv_provider import CSVProvider, _to_number, normalize_prices

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices"


def test_to_number_handles_vietnamese_formats():
    s = pd.Series(["1,234,500", "1.234.500", "98,5", "98.5", "12"])
    assert _to_number(s).tolist() == [1234500, 1234500, 98.5, 98.5, 12]


def test_normalize_vietnamese_headers_and_units():
    raw = pd.DataFrame({
        "Ngày": ["03/01/2025", "02/01/2025"],
        "Giá mở cửa": [10.5, 10.0],
        "Giá cao nhất": [10.8, 10.6],
        "Giá thấp nhất": [10.4, 9.9],
        "Giá đóng cửa (nghìn VNĐ)": [10.7, 10.5],
        "KL khớp lệnh": ["1,000,000", "800,000"],
    })
    df = normalize_prices(raw, "abc")
    assert df["symbol"].unique().tolist() == ["ABC"]
    assert df["date"].is_monotonic_increasing                     # đã sắp xếp lại
    assert df["date"].iloc[0] == pd.Timestamp("2025-01-02")       # dd/mm, không phải mm/dd
    assert df["close"].tolist() == [10_500, 10_700]               # nghìn đồng -> đồng
    assert df["volume"].tolist() == [800_000, 1_000_000]


def test_missing_column_gives_clear_error():
    with pytest.raises(ValueError, match="thiếu cột"):
        normalize_prices(pd.DataFrame({"date": ["2025-01-02"], "close": [1]}), "X")


def _clean_frame():
    return pd.DataFrame({
        "symbol": "X",
        "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        "open": [100.0, 101, 102], "high": [102.0, 103, 104],
        "low": [99.0, 100, 101], "close": [101.0, 102, 103], "volume": [10, 10, 10],
    })


def test_validate_clean_data_has_no_issues():
    assert validate_prices(_clean_frame()) == []


def test_validate_catches_errors_and_jumps():
    df = _clean_frame()
    df.loc[1, "high"] = 50          # high < close -> lỗi
    df.loc[2, "close"] = 130        # +27% -> vượt biên độ
    df.loc[2, "high"] = 131
    rules = {i.rule for i in validate_prices(df)}
    assert "high_nho_hon_open_close" in rules
    assert "vuot_bien_do" in rules


@pytest.mark.skipif(not SAMPLE.exists(), reason="chạy scripts/make_sample_data.py trước")
def test_sample_files_load_and_pass_validation():
    prov = CSVProvider(SAMPLE)
    assert prov.symbols() == ["DEMO_A", "DEMO_B", "DEMO_C"]
    for sym in prov.symbols():
        df = prov.ohlcv(sym)
        assert len(df) > 1000
        errors = [i for i in validate_prices(df) if i.level == "LOI"]
        assert errors == [], (sym, errors)


def test_store_roundtrip(tmp_path):
    pytest.importorskip("duckdb")
    from vnsa.data.store import Store

    with Store(tmp_path / "t.duckdb") as store:
        store.write_prices(_clean_frame(), "test")
        store.write_prices(_clean_frame(), "test")  # nạp lại không bị trùng
        out = store.prices("x")
        assert len(out) == 3
        assert out["close"].tolist() == [101, 102, 103]
        assert store.inventory()["so_phien"].tolist() == [3]


def test_adjusted_price_scales_whole_candle():
    raw = pd.DataFrame({
        "Ngày": ["02/01/2025"], "Giá đóng cửa": [100.0], "Giá điều chỉnh": [90.0],
        "Giá mở cửa": [98.0], "Giá cao nhất": [102.0], "Giá thấp nhất": [97.0],
        "Khối lượng khớp lệnh": [1000],
    })
    row = normalize_prices(raw, "X").iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (88_200, 91_800, 87_300, 90_000)


def test_import_excel_merges_files_like_cafef(tmp_path):
    pytest.importorskip("openpyxl")
    from vnsa.data.excel_import import import_excel_folder

    cols = ["Mã", "Ngày", "Giá đóng cửa", "Giá điều chỉnh", "Thay đổi", "Khối lượng khớp lệnh",
            "Giá trị khớp lệnh", "Khối lượng thỏa thuận", "Giá trị thỏa thuận",
            "Giá mở cửa", "Giá cao nhất", "Giá thấp nhất"]
    folder = tmp_path / "excel" / "fpt"          # thư mục chữ thường vẫn nhận
    folder.mkdir(parents=True)
    a = [["FPT", "03/01/2025", 51, 51, "+1", 900, 0, 5, 0, 50, 51.5, 49.8],
         ["FPT", "02/01/2025", 50, 50, "0", 800, 0, 0, 0, 49.5, 50.5, 49.0]]
    b = [["FPT", "06/01/2025", 52, 52, "+1", 700, 0, 0, 0, 51, 52.5, 50.9],
         ["FPT", "03/01/2025", 51, 51, "+1", 900, 0, 5, 0, 50, 51.5, 49.8]]   # trùng ngày
    pd.DataFrame(a, columns=cols).to_excel(folder / "nam1.xlsx", index=False)
    pd.DataFrame(b, columns=cols).to_excel(folder / "nam2.xlsx", index=False)

    out, df, files = import_excel_folder("FPT", tmp_path / "excel", tmp_path / "prices")
    assert len(files) == 2 and out.exists()
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03", "2025-01-06"]
    assert df["volume"].tolist() == [800, 900, 700]   # khối lượng KHỚP LỆNH, không phải thỏa thuận
    assert df["close"].tolist() == [50_000, 51_000, 52_000]
    assert validate_prices(df) == []
