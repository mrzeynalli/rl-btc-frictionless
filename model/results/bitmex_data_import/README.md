# BitMEX OHLCV-only import and cleanup

Active workflow: `collector.bitmex_ohlcv`, independent of the trading agent.
Service: `cryptorl-bitmex-ohlcv-download.service`.
Log: `ohlcv.log`.

Dataset progress and manifest:
`../../data/bitmex_xbtusd_1m_2016_2026_08_01/status.json` and `manifest.json`.
The market file is `XBTUSD-1m-OHLCV.parquet`, with six columns including UTC time.
After the user clarified that futures funding is needed, the public funding
history was restored as `funding/XBTUSD-funding.parquet`. It is 143 KB and has
11,153 audited settlement records; its temporary API response pages were removed.

The previous extended importer retained roughly 26 GB of raw trades and failed
on nonfinite auxiliary index data. It was stopped, not merely left retrying.
Its 1,848 successfully verified daily outputs were reused. The new job does
not fetch funding or index data and does not persist raw trades.

`ohlcv_cleanup.json` records permanent deletion of raw/auxiliary data, download
caches and smoke copies. Duplicate daily bar shards and the two explicitly
retired Binance Spot datasets are removed after the full OHLCV audit passes.
Nothing is moved to the trash. No data was found in the standard root-user or
root-filesystem trash locations inspected during cleanup. Existing Git history
and unrelated user files are untouched.

Nine importer/cleanup tests passed before launch. Launch-period, modern-schema
and missing-archive smoke checks had passed previously. June 10, 2026 contributes
1,437 actual-trade bars recovered via the public REST API; the raw recovery
records have now been removed. `download.log` preserves the original failed
job's history, and `ohlcv.log` is the current job's progress log.

June 26/30, 2026 also contain match identifiers reused across distinct regular
fills and spread legs. The ID-only validation was corrected to preserve
distinct fills and check complete regular records. A regression test covers
volume preservation. June 30 09:13 UTC has 61,800 regular-trade contracts;
BitMEX's bucketed API adds 5,500 spread-leg contracts to report 67,300. The
dataset deliberately excludes those special prints and documents that policy.

July 15 13:02 UTC contains two identical published regular records. Keeping
both yields 154 total prints / 914,400 contracts, exactly matching the native
13:03 bucket; removing one incorrectly subtracts 100 contracts. Records are
therefore not deduplicated by fields either. This behavior is regression-tested.
The final minute timestamps themselves must always be unique.
