# VN Stock Analyzer — ghi chú cho Claude Code

Tool phân tích và dự báo cổ phiếu Việt Nam, dùng cá nhân. Người dùng học tài chính,
mới làm quen Python/VS Code: giải thích ngắn gọn bằng tiếng Việt, đưa lệnh cụ thể.

## Môi trường
- Chính: GitHub Codespaces (Linux, Python 3.12, cấu hình ở `.devcontainer/`). Thư viện đã cài sẵn
  khi tạo Codespace; không cần `.venv`.
- Phụ: máy Windows của người dùng (PowerShell, Python 3.13, môi trường ảo `.venv`; đường dẫn có
  dấu cách và chữ có dấu — luôn đặt trong ngoặc kép).
- Cài/cập nhật: `pip install -e ".[dev,mcp,dashboard]"`. Test: `pytest`.
- `data/raw/` và `*.duckdb` không đưa lên Git (xem `.gitignore`): dữ liệu thật phải tải lại trong
  từng môi trường (`python -m vnsa.cli fetch`, hoặc vnstock: `python scripts/fetch_vnstock.py FPT HPG MWG VNM DGC --years 5`).
- vnstock cài từ kho gói của nhà phát triển (PyPI đang cách ly gói này):
  `pip install -U --extra-index-url https://vnstocks.com/api/simple vnstock vnai`.

## Kiến trúc (giữ nguyên nguyên tắc)
- Mọi nguồn dữ liệu là lớp con của `DataProvider` (`src/vnsa/data/providers/base.py`).
- Phân tích chỉ đọc từ kho DuckDB `data/warehouse.duckdb` qua `Store`, không gọi nguồn trực tiếp.
- Giá lưu bằng **đồng**, đã **điều chỉnh** cổ tức/chia tách; khối lượng là **khớp lệnh**.
- Cấu hình (biên độ, danh sách mã, đường dẫn) nằm ở `config/settings.toml`, không viết cứng trong code.
- Mọi con số do Python tính; LLM chỉ diễn giải. Không bịa dữ liệu thị trường.

## Lệnh
- `python -m vnsa.cli analyze MÃ... [--brief] [--refresh] [--full] [--offline]` — **mã bất kỳ** HOSE/HNX/
  UPCoM: `data/autofetch.py` tự tải phần thiếu/cũ (sàn+ngành và BCTC qua vnstock, giá CafeF tải tăng dần — CafeF
  không trả một số mã HNX/UPCoM nên tự dự phòng sang vnstock `Quote` (VCI),
  tự tải lại toàn bộ khi CafeF điều chỉnh giá cũ) rồi in toàn bộ phân tích. Tham số `[auto]`.
  Biên độ/bước giá theo sàn qua `src/vnsa/market.py` (bảng `companies`; chưa biết sàn → HOSE), không
  dùng thẳng `band_hose`. Ngân hàng/CK/bảo hiểm (`model` ≠ general): `analysis.fa/valuation` báo
  ValueError; dùng `analysis.financial` (RI + P/B theo ROE, điểm FA riêng có chỉ số ngành). MCP: `analyze_symbol`, `update_symbol` (chuyển
  stdout sang stderr khi tải). Dashboard: ô "Mã bất kỳ" ở thanh bên.
- Khuyến nghị HAI TRỤC (`recommend/engine.py`, `[recommend]`): xếp hạng MUA/KHẢ QUAN/NẮM GIỮ/KÉM KHẢ
  QUAN/BÁN theo lợi nhuận kỳ vọng = upside giá trị hợp lý + cổ tức 12 tháng (ngưỡng 15/5/−5/−15%);
  rào chắn: VNINDEX giảm (`ta/regime.py`) nâng ngưỡng 5% và giảm rủi ro/lệnh 50%, FA < 50 hạ một bậc,
  Altman Z'' < 1,1 tối đa NẮM GIỮ; thiếu giá trị hợp lý = KHÔNG XẾP HẠNG. Trục THỜI ĐIỂM theo
  `[recommend] timing_rule` = "none" (chỉ vùng mua + cắt lỗ) vì kiểm định 10 năm × 45 mã
  (`study-timing`, `recommend/study.py`, docs/kiem_dinh_thoi_diem.md) không quy tắc nào thắng mua &
  giữ cả hai giai đoạn; "ta" (cũ) hoặc quy tắc trong `recommend/strategies.py` (`rule_state`). Điểm FA/TA chỉ để sắp xếp; Quant chỉ tham khảo. Định giá trong điểm FA là TƯƠNG ĐỐI
  (P/E, P/B ÷ trung vị lịch sử của mã: `pe_rel`, `pb_rel`). Khối lượng chặn bởi trần 15% vốn/mã
  và 10% GTGD bình quân 20 phiên (`[plan]`).
- `analyze` mặc định in THẺ BÁO CÁO (`report/card.py`, `--detail` để in thêm chi tiết): mọi số lấy
  từ `analysis.assess()` (tính một lần: FA, giá trị hợp lý kèm P10/P90 kịch bản, TA, vùng mua, rủi
  ro, điểm Quant, khuyến nghị, khối ngoại/tự doanh từ bảng `flows` — chỉ tham khảo). Điểm sắp xếp =
  FA 60% + TA 40%; Quant `[recommend.quant]` chỉ tham khảo; độ tin cậy 6 điều kiện (`[report]`; "Cao"
  cần kịch bản hẹp và upside ≤ max_abs_upside); giá trị hợp lý = bình quân có trọng số
  `[valuation.weights]`, loại phương pháp lệch > 2,5 lần trung vị. MCP `analyze_symbol` trả `report`.
- Số liệu hiện tại: `fa/ttm.py` (`Data.current`) = TTM từ 4 quý gần nhất (bảng `fundamentals_q`) nếu
  mới hơn BCTC năm; EPS/BVPS chia số cp lưu hành hiện tại (`companies.shares_out` từ Company.overview,
  vốn hoá / giá vì VCI trả `issue_share` trùng tên). P/E, P/B lịch sử theo vốn hoá thật
  (`prices.close_raw` × vốn góp năm đó / mệnh giá). Cổ tức: bảng `dividends` (Company.events).
  `scripts/probe_vnstock.py` + input `probe` của workflow để dò cấu trúc dữ liệu vnstock mới.
- Giá mua: `ta/entry.py` vùng mua (neo hỗ trợ/VWAP20/SMA20, chặn bởi giá trị hợp lý × (1 − biên an
  toàn), "CHỜ GIÁ" nếu quá cao; `[entry]`); `analysis.plan` dùng giữa vùng làm giá vào, KHÔNG dùng
  giá đóng cửa. Giá trị hợp lý `fa/fair_value.py` = bình quân DCF + P/E, P/B lịch sử (EPS/BVPS theo
  số cp hiện tại; bội số mục tiêu = hiện tại + `multiple_reversion` × (trung vị − hiện tại),
  `fv.revert`; `pe_median`/`pb_median` là trung vị thật, dùng cho điểm FA tương đối; P/E, P/B ngành
  `analysis.peer_multiples` khi kho có ≥ `min_peers` mã cùng ngành + loại DN; độ tin cậy "Cao" cần
  phương pháp độc lập (`FairValue.independent`) và lệch ≤ 30% giá mục tiêu CTCK `analyst_target`;
  cổ tức > `max_dividend_yield` dùng bình quân 3 năm; F-Score phát hành theo vốn góp mẹ trừ thưởng;
  chất lượng lợi nhuận `fv.quality_adjust`: P/B lịch sử × ROE hiện tại/ROE trung vị (≤ 1,
  `roe_adjust`), LNST giảm > 5%/năm thì P/E mục tiêu ≤ P/E hiện tại và DCF lấy FCFF năm cuối
  (`no_rerate_below`, `declining_base`); giá trị hợp lý chặn ở giá mục tiêu CTCK × 1,3
  (`consensus_cap`, `fv.cap_by_consensus`, giá trị mô hình giữ ở `FairValue.uncapped`)); định chế tài chính: thu nhập thặng dư + P/B theo ROE (`analysis.financial`,
  `scores.financial_score`, `[valuation.financials]`). Khuyến nghị tự lưu vào bảng `signals` +
  `data/signals.csv` (`analysis.record_signals`); `review-signals` chấm sau 5/20/60 phiên
  (`recommend/review.py`); test cô lập CSV bằng `VNSA_SIGNALS_CSV`.
- `python -m vnsa.cli fetch [MÃ...] [--years 5]` — tải giá từ CafeF (API JSON không chính thức,
  ≤1 năm mỗi lần gọi, ngày dạng tháng/ngày/năm) → `data/raw/prices/` → kho. Cách chính để lấy giá.
- `python -m vnsa.cli fetch-bctc [MÃ...]` — BCTC năm hợp nhất qua vnstock nguồn VCI → `data/raw/bctc/`
  và bảng `fundamentals` trong kho. Bản miễn phí chỉ 4 năm gần nhất. KHÔNG dùng nguồn KBS (nhãn năm
  bị đảo ngược). `python -m vnsa.cli ratios FPT` — bộ chỉ số (`src/vnsa/fa/ratios.py`, quy ước ở đầu file).
- `python -m vnsa.cli value FPT [--growth 0.1] [--wacc 0.11]` — WACC (beta tuần so với VNINDEX), DCF-FCFF
  hai giai đoạn (g1 giảm dần về g), P/E, P/B, ma trận độ nhạy. Giả định ở `[valuation]` trong settings.
- `fa-score FPT` (F-Score, Altman Z, điểm FA) · `ta FPT` (chỉ báo, hỗ trợ/kháng cự, điểm TA) ·
  `plan FPT` (cắt lỗ ATR, mục tiêu, khối lượng theo % rủi ro, bước giá HOSE) · `risk FPT [--days N]`
  (GARCH(1,1) tự viết bằng scipy, Monte Carlo lọc lịch sử chặn biên độ ±7%, VaR/CVaR).
- `recommend [MÃ...]` (bảng xếp hạng hai trục) · `backtest FPT` (`analysis.backtests` qua
  `strategies.simulate`: điểm TA, phá nền 55 phiên, quy tắc đang dùng; khớp mở cửa phiên sau, không
  mua ở trần/không bán khi nằm sàn, T+2, phí + thuế) · `study-timing` (walk-forward nhiều mã,
  `[study]`, workflow `study.yml`). Mọi phép tính dùng chung qua `src/vnsa/analysis.py`.
- Định chế tài chính: chỉ số ngành `fa/sector.py` (NIM xấp xỉ, CIR, chi phí tín dụng, LDR đơn giản;
  ký quỹ/vốn chủ cho CK) từ khoản mục `SECTOR_FIELDS` (`fa/ratios.py`, đọc theo item_id VCI ở
  `SECTOR_MAPPING`, không báo thiếu với doanh nghiệp thường). NPL, CASA, CAR KHÔNG có trong nguồn
  miễn phí: báo thiếu, không ước lượng — người dùng nhập tay `config/bank_manual.toml`
  (`sector.manual_bank`, nhóm điểm `chat_luong_tai_san`, CASA, CAR). NIM/CIR/chi phí tín dụng chấm
  so với trung vị ngân hàng trong kho khi ≥ `min_bank_peers` (`analysis.bank_peers`,
  `[fa_score.financials.relative]`). Dò khoản mục: `python scripts/probe_vnstock.py --items VCB`.
- Lịch sử bền vững (`history.py`): workflow `.github/workflows/daily.yml` (16:40 giờ VN, ngày giao
  dịch) phân tích `[signals] daily`, ghi `history/{signals,fundamentals,fundamentals_q}.csv` + `review.txt`
  vào nhánh `signals`. `python -m vnsa.cli history import --git origin/signals` nạp về kho (chỉ thêm
  dòng chưa có, BCTC nạp vào đánh dấu cũ để autofetch vẫn tải mới); `history export DIR`. BCTC năm
  ghi đè theo (mã, năm) nên năm cũ hơn 4 năm miễn phí được giữ lại (tích luỹ).
- Lớp vĩ mô (`src/vnsa/macro/`, CHỈ THAM KHẢO, chưa vào điểm): `fetch-macro` (Yahoo: USD/VND,
  USD/JPY, ^TNX, dầu, DXY, quặng sắt, HRC, bông, đường; World Bank Pink Sheet: urê, DAP, đá phốt phát,
  than, cao su; FRED cần User-Agent curl; JPY/VND = derived) → bảng `macro_series`; số Việt Nam NHẬP
  TAY `config/macro_manual.toml` (GSO không truy cập được từ Actions). Danh mục + nhóm bối cảnh +
  ánh xạ ngành (`[[industry]]` regex ICB cấp 4→3→2, `[symbol.MÃ]`) ở `config/macro_map.toml`.
  `macro/signals.py` biến động + z 5 năm; `macro/score.py` điểm −1..+1 (độ phủ < 50% = chưa đủ số);
  `macro/sensitivity.py` hồi quy tháng mã ~ VNINDEX + yếu tố (≥ 36 tháng) báo NGƯỢC dấu. Thẻ báo cáo
  mục VĨ MÔ & NGÀNH; rào chắn: ngành ≤ −0,5 (độ phủ ≥ 60%) chỉ giảm độ tin cậy một bậc; k_e/WACC
  dùng `analysis.risk_free()` (TPCP 10 năm nhập tay nếu ≤ 45 ngày); `signals.macro_industry/
  macro_market` + `review-signals` chia theo nhóm vĩ mô. Tin tức: tiêu đề `Company.news` (VCI) →
  bảng `news`. Lệnh `macro [MÃ...]`, MCP `macro_context`, tab dashboard "Vĩ mô", workflow `macro.yml`
  (input probe: `scripts/probe_macro.py`). vnstock trả ngành dạng dài (icb_level/icb_name):
  `_wide_icb` đổi về icb_name1..4; `companies.industry` = cấp 2, `industry3/4` cho vĩ mô.
- Sàng lọc toàn thị trường (`src/vnsa/screen/`, `[screen]`, lệnh `screen [--stage liquidity|industry]
  [--groups]`, workflow `screen.yml` có cache kho): T0 vũ trụ (`Store.companies`); T1 thanh khoản
  (`screen/liquidity.py`: GTGD 20 phiên `close_raw` × KL, số phiên, ngừng GD, RS 3/6 tháng, MA50/200) — giá
  1 năm qua `Updater.quick_prices`, `ensure` tự bổ sung đủ `[auto] years` năm (`_short`, bảng `price_depth`;
  tải lại vì điều chỉnh giá giữ độ sâu `_full_start`); T2 (`screen/deep.py`) hai lượt: `fetch_all` (ensure
  không khối ngoại/tin, thanh khoản cao trước, dừng sau `deep_minutes`) rồi `assess_all` (assess + build_card
  → cột bảng `screen`; tăng trưởng LNST: 8 quý cùng kỳ > TTM so năm > năm so năm); T3 (`screen/industry.py`)
  nhóm theo `macro.score.match_industry` (`screen/groups.py`, rơi về ICB cấp 2), 6 thành phần đổi ra thứ hạng
  0–100 giữa nhóm, thiếu thì chia lại trọng số, < `min_members` mã không xếp; bảng `screen_industry`.
  T4 (`screen/pick.py`): trong `top_industries` nhóm, loại ≠ MUA/KHẢ QUAN, FA < `min_fa`, Z'' < `min_z`, cờ
  đỏ, tin cậy < `min_confidence`; xếp MUA > KHẢ QUAN > điểm > RS 6T; `per_industry` mã (stage 4,
  `pick_rank`); MUA đạt ngoài nhóm dẫn đầu = `outside_picks`. Báo cáo `screen/report.py format_picks`;
  dashboard tab "Sàng lọc thị trường", MCP `screen_market` (đọc lượt mới nhất). T1 tải song song
  (`Updater._prefetch`, `[screen] workers`, ghi kho tuần tự). vnstock: mọi lượt gọi qua
  `vnstock_provider._call` (`RateLimiter` theo `[vnstock]`, khoá `VNSTOCK_API_KEY`) vì vnai vượt hạn mức
  (Khách 20/phút, 1.200/giờ) là SystemExit. Kiểm chứng: `review-screen` (`screen/review.py`: lượt đã lưu, lãi/lỗ
  20/60 phiên theo tầng 4 / ≥3 / ≥1, bình quân theo lượt) và `study-screen` (`screen/study.py`: walk-forward
  tháng 10 năm phần giá — RS + độ rộng nhóm, RS 6T trong nhóm, phí/thuế [backtest]; thiên lệch sống sót).
  Lịch sử: `history export/import` thêm `screen.csv` (tầng ≥ 1), `screen_industry.csv` (nạp theo lượt).
  Workflow `screen.yml`: thứ Bảy 8:17 giờ VN, ghi nhánh `signals` (chỉ nhánh mặc định).
- MCP server: `python -m vnsa.mcp_server` (extra `[mcp]`, chạy được mcp 1.x và 2.x); cấu hình mẫu cho
  Claude Desktop ở `docs/`. Dashboard: `streamlit run dashboard/app.py` (extra `[dashboard]`, cổng 8501).
- `python scripts/smoke_test.py` gọi mọi công cụ MCP + dashboard trên kho thật. Test dùng fixture
  `warehouse` (tests/conftest.py) và biến môi trường `VNSA_WAREHOUSE` để không đụng kho thật.
- Nâng cấp v2.0 (nhánh `nang-cap-v2`): kho dùng chung nhiều tiến trình — `store.connect` chờ khi bận
  (`WarehouseBusy`), `Store.lazy` (LazyStore) mở kho theo từng lời gọi trong autofetch/vĩ mô, tải hàng
  loạt mở lại kho mỗi 25 mã (`_batched`); bảng `meta` giữ `SCHEMA_VERSION` (thêm ALTER viết tay thì
  tăng `SCHEMA_REVISION`); KHÔNG dùng `fetchone()` (DuckDB 1.4 báo "unfinished result") — dùng
  `Store._first`. `config.validate` (trọng số ≥ 0, ngưỡng đúng chiều theo `LOWER_IS_BETTER`, không bắt
  tổng = 100). BCTC theo mùa `autofetch.bctc_due/expected_period` (`[auto] bctc_*`). Định giá theo
  nhóm `analysis.valuation_group` (`[valuation.groups]`). `strategies.rule_set` có `vcp_pullback`
  (`bb_width_pct` = thứ hạng 120 phiên của độ rộng Bollinger). `[risk] drift = "zero"`.
  Sổ lệnh `portfolio.py` (CSV `[data] trades`, gitignore — repo công khai; test dùng `VNSA_TRADES`),
  lệnh `portfolio`, MCP `get_portfolio`; `analysis.plan` gọi `portfolio.sizing_limits` (tương quan,
  `[portfolio] max_sector_pct`). Chỉ hiển thị, chưa vào điểm: `ttm.earnings_acceleration`,
  `flow_summary` `smart_10`, độ nén giá; `[regime] early` (tắt).
- Nâng cấp v2.1 (nhánh `nang-cap-v2.1`): thang mua 3 lớp `ta/entry.py` (`build_ladder`, `Tier`,
  `EntryZone.tiers/stop_cap`; low = lớp 3, high = lớp 1, mid = bình quân có trọng số; `[entry] ladder*`),
  `trade_plan(max_stop=...)` đặt cắt lỗ dưới lớp 3, `analysis._fill_probs`/`Simulation.prob_touch` xác suất
  khớp; kiểm định `study-entry` (`recommend/entry_study.py`, workflow `entry.yml`). Sổ lệnh: sự kiện quyền
  `positions(..., actions=corporate_actions(...))` (cổ tức tiền hạ cắt lỗ, thưởng cp cộng SL), sao lưu
  `portfolio backup/restore` (`VNSA_BACKUP_REPO`, `VNSA_BACKUP_TOKEN`, chặn repo công khai), nhiệt danh mục
  `portfolio_heat` (`[portfolio] max_heat`; extra_caps dạng ("risk"|"value", đồng, lý do)). Quy tắc
  `breakout_early` (`rule_set(..., regime_early)`, `data["__regime_early__"]`).
- `python -m vnsa.cli import-excel FPT` — ghép `data/raw/excel/FPT/*.xlsx` (CafeF) → CSV → kho
- `python -m vnsa.cli load [--sample]` · `check FPT` · `list` · `chart FPT`

## Lộ trình sprint 10 ngày (đã xong cả 10 ngày)
1. Dữ liệu giá ✔ — 5 năm cho FPT, HPG, MWG, VNM, DGC (lệnh `fetch`)
2. BCTC + bộ chỉ số tài chính ✔ (4 năm 2022–2025; còn đối chiếu thêm với BCTC gốc)
3. WACC, DCF-FCFF, P/E, P/B, ma trận độ nhạy ✔ (lệnh `value`; giả định cần người dùng duyệt)
4. F-Score, Altman Z, điểm FA; chỉ báo TA ✔ (`fa-score`, `ta`)
5. Hỗ trợ/kháng cự, điểm TA, kế hoạch giao dịch (cắt lỗ theo ATR) ✔ (`ta`, `plan`)
6. GARCH, Monte Carlo P10–P90, VaR ✔ (`risk`)
7. Engine khuyến nghị + backtest đơn giản ✔ (`recommend`, `backtest`)
8. MCP server cho Claude Desktop ✔ (`src/vnsa/mcp_server.py`)
9. Dashboard Streamlit ✔ (`dashboard/app.py`)
10. Dự phòng, sửa lỗi ✔ (`scripts/smoke_test.py`, chạy thật trên GitHub Actions)

Mỗi bước: viết test cho công thức tài chính, chạy `pytest` trước khi báo xong.
Chạy thật trên máy chủ GitHub: workflow `.github/workflows/data.yml` (Actions → Run workflow; tuỳ chọn
bỏ qua bước tải giá). Kết quả in trong log, file CSV/kho ở mục Artifacts.
