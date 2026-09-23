"""Offline tests of the BitMEX archive importer."""
import numpy as np
import pandas as pd
import pytest

from data_pipeline.collector.bitmex import LAUNCH, combine_parts, minute_parts, normalize_trades, udf_frame, with_missing_days


def sample():
    return pd.DataFrame({
        "timestamp": ["2016-05-14D00:01:00.000000000", "2016-05-14D00:00:59.999999999",
                      "2016-05-14D00:00:01.000000000", "2016-05-14D00:00:01.000000000"],
        "symbol": ["XBTUSD"] * 4, "side": ["Buy", "Sell", "Buy", "Sell"],
        "size": [100, 200, 100, 300], "price": [500., 400., 450., 460.],
        "grossValue": [20_000_000, 50_000_000, 22_222_222, 65_217_391],
        "homeNotional": [0.2, 0.5, 100/450, 300/460], "foreignNotional": [100., 200., 100., 300.],
        "source_row": [0, 1, 2, 3], "trdMatchID": ["a", "b", "c", "d"]})


def test_inverse_units_and_raw_timestamp_precision():
    frame = normalize_trades(sample(), pd.Timestamp("2016-05-14", tz="UTC"))
    assert frame.timestamp.iloc[1].nanosecond == 999
    assert frame.contract_regime.eq("inverse_perpetual").all()
    wrong = sample()
    wrong.loc[0, "foreignNotional"] = 99
    with pytest.raises(ValueError, match="notional"):
        normalize_trades(wrong, pd.Timestamp("2016-05-14", tz="UTC"))


def test_minute_boundaries_causal_ohlc_and_chunk_replay():
    frame = normalize_trades(sample(), pd.Timestamp("2016-05-14", tz="UTC"))
    full = combine_parts([minute_parts(frame)])
    split = combine_parts([minute_parts(frame.iloc[:2]), minute_parts(frame.iloc[2:])])
    pd.testing.assert_frame_equal(full, split)
    assert len(full) == 2
    assert full.iloc[0].open == 450
    assert full.iloc[0].close == 400
    assert full.iloc[0].high == 460
    assert full.iloc[0].volume_contracts == 600
    assert full.iloc[0].taker_buy_contracts == 100
    assert full.iloc[0].taker_sell_contracts == 500
    assert full.iloc[1].open == 500  # actual first trade, NOT previous close
    assert (full.last_trade_time < full.close_time).all()


def test_prelaunch_same_symbol_is_not_inverse_history():
    frame = sample().iloc[:1].copy()
    frame["timestamp"] = "2016-05-13D11:59:59.999999999"
    frame["foreignNotional"] = 9000.  # old contract units must not enter inverse bars
    result = normalize_trades(frame, LAUNCH.floor("D"))
    assert result.contract_regime.iloc[0] == "pre_launch_legacy"
    assert not (result.timestamp >= LAUNCH).any()


def test_reference_timestamps_and_week_overlap():
    start = pd.Timestamp("2026-08-01", tz="UTC")
    stamp = int(start.timestamp())
    payload = {"s": "ok", "t": [stamp-60, stamp, stamp+60],
               "o": [1, 2, 3], "h": [2, 3, 4], "l": [2, 3, 4],
               "c": [2, 3, 4], "v": [0, 0, 0]}
    frame = udf_frame(payload, start, start + pd.Timedelta(minutes=1))
    assert len(frame) == 1
    assert frame.iloc[0].available_at == start + pd.Timedelta(minutes=1)
    assert frame.iloc[0].close == 3
    assert udf_frame({"s": "no_data"}, start, start).empty


def test_no_fill_of_minutes_without_trades():
    frame = normalize_trades(sample().iloc[:1], pd.Timestamp("2016-05-14", tz="UTC"))
    bars = combine_parts([minute_parts(frame)])
    assert len(bars) == 1
    assert bars.iloc[0].open_time.minute == 1
    np.testing.assert_allclose(bars.volume_btc, bars.taker_buy_btc + bars.taker_sell_btc)


def test_missing_archive_is_explicit_and_recovered_not_filled():
    start = pd.Timestamp("2026-06-09", tz="UTC")
    end = pd.Timestamp("2026-06-12", tz="UTC")
    records = [{"day": "2026-06-09", "key": "first"}, {"day": "2026-06-11", "key": "last"}]
    result = with_missing_days(records, start, end)
    assert len(result) == 3
    assert result[1]["day"] == "2026-06-10"
    assert result[1]["source"] == "public_rest_trade_fallback"
    assert with_missing_days(result, start, end) == result
