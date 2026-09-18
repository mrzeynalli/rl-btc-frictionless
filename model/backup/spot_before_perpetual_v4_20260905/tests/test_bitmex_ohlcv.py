import pandas as pd
import pytest

from collector.bitmex_ohlcv import COLUMNS, compact_frame, consolidate, remove_exact


def test_ohlcv_projection_and_contract_volume():
    frame = pd.DataFrame({"open_time": pd.date_range("2020-01-01", periods=2, freq="min", tz="UTC"),
                          "open": [10., 11.], "high": [12., 12.], "low": [9., 10.],
                          "close": [11., 10.], "volume_contracts": [100, 200], "unneeded": [9, 9]})
    result = compact_frame(frame)
    assert list(result) == COLUMNS
    assert result.volume.tolist() == [100, 200]
    assert str(result.open_time.dt.tz) == "UTC"
    assert str(result.open_time.dtype) == "datetime64[us, UTC]"
    frame.loc[0, "high"] = 8
    with pytest.raises(ValueError, match="Invalid OHLCV"):
        compact_frame(frame)


def test_timestamp_precision_is_normalized_across_shards():
    base = {"open": [1.], "high": [1.], "low": [1.], "close": [1.],
            "volume_contracts": [1]}
    nano = compact_frame(pd.DataFrame({"open_time": pd.Series(
        [pd.Timestamp("2020-01-01", tz="UTC")], dtype="datetime64[ns, UTC]"), **base}))
    micro = compact_frame(pd.DataFrame({"open_time": pd.Series(
        [pd.Timestamp("2020-01-02", tz="UTC")], dtype="datetime64[us, UTC]"), **base}))
    assert nano.dtypes.equals(micro.dtypes)


def test_delete_is_allowlisted_permanent_and_audited(tmp_path):
    extra = tmp_path / "raw_trades"
    extra.mkdir()
    (extra / "sample").write_bytes(b"12345")
    protected = tmp_path / "keep.parquet"
    protected.write_bytes(b"keep")
    report = tmp_path / "cleanup.json"
    with pytest.raises(ValueError, match="Unsafe cleanup"):
        remove_exact(protected, [extra], report)
    assert remove_exact(extra, [extra], report) == 5
    assert not extra.exists()
    assert protected.exists()
    assert report.exists()


def test_funding_is_not_in_ohlcv_cleanup_targets():
    from collector.bitmex_ohlcv import PRUNED_DATA_NAMES
    assert "funding" not in PRUNED_DATA_NAMES


def test_ohlcv_download_never_persists_raw_trades_and_resumes_without_them(tmp_path, monkeypatch):
    import hashlib
    from collector import bitmex as source
    cache = tmp_path / "download_cache/20260801.csv.gz"
    cache.parent.mkdir()
    raw = pd.DataFrame({"timestamp": ["2026-08-01D00:00:01.000000"],
                        "symbol": ["XBTUSD"], "side": ["Buy"], "size": [100],
                        "price": [50000.0], "grossValue": [200000],
                        "homeNotional": [0.002], "foreignNotional": [100.0],
                        "trdMatchID": ["unique-test-trade"]})
    raw.to_csv(cache, index=False, compression="gzip")
    item = {"day": "2026-08-01", "bytes": cache.stat().st_size,
            "etag": hashlib.md5(cache.read_bytes()).hexdigest(), "key": "test"}
    def disallow_network(*args, **kwargs):
        raise AssertionError("A cached or completed day must not access the network")
    monkeypatch.setattr(source, "get", disallow_network)
    result = source.collect_day(item, tmp_path, source.START, source.END, keep_raw=False)
    assert result["bars"] == 1
    assert not (tmp_path / "trades").exists()
    assert not cache.exists()
    assert list(result["outputs"]) == ["bars_1m/20260801.parquet"]
    resumed = source.collect_day(item, tmp_path, source.START, source.END, keep_raw=False)
    assert resumed["outputs"] == result["outputs"]
    monkeypatch.setattr(source, "START", pd.Timestamp("2026-08-01", tz="UTC"))
    monkeypatch.setattr(source, "LAUNCH", source.START)
    manifest = consolidate(tmp_path, [resumed])
    restored = pd.read_parquet(tmp_path / manifest["file"])
    assert manifest["audit_passed"]
    assert manifest["rows"] == 1
    assert list(restored) == COLUMNS
    assert restored.iloc[0].volume == 100
    assert restored.iloc[0].open == 50000
    assert str(restored.open_time.dt.tz) == "UTC"


@pytest.mark.parametrize("identical", [False, True])
def test_distinct_fills_with_reused_match_ids_keep_volume(tmp_path, identical):
    import hashlib
    from collector import bitmex as source
    cache = tmp_path / "download_cache/20260801.csv.gz"
    cache.parent.mkdir()
    raw = pd.DataFrame({"timestamp": ["2026-08-01D00:00:01.000000"] * 2,
                        "symbol": ["XBTUSD"] * 2, "side": ["Buy"] * 2,
                        "size": [100, 200], "price": [50000., 50001.],
                        "grossValue": [200000, 399992],
                        "homeNotional": [0.002, 0.00399992], "foreignNotional": [100., 200.],
                        "trdMatchID": ["same-matcher-id"] * 2})
    if identical:
        raw.iloc[1] = raw.iloc[0]
    raw.to_csv(cache, index=False, compression="gzip")
    item = {"day": "2026-08-01", "bytes": cache.stat().st_size,
            "etag": hashlib.md5(cache.read_bytes()).hexdigest(), "key": "test"}
    result = source.collect_day(item, tmp_path, source.START, source.END, keep_raw=False)
    bars = pd.read_parquet(tmp_path / "bars_1m/20260801.parquet")
    assert result["reused_match_identifiers"] == 1
    assert bars.iloc[0].volume_contracts == (200 if identical else 300)
    assert bars.iloc[0].trades == 2
    assert result["repeated_regular_records"] == int(identical)
