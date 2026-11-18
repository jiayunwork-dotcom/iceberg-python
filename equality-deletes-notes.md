# Iceberg v2 equality delete reads — implementation notes and verification

This document accompanies the PyIceberg 0.10.0 change that adds bounded-memory
reading of Iceberg **format v2** Parquet equality delete files on local tables.
It records the design decision, the memory model, and the reproducible
acceptance results.

## Scope

Supported:

- Format **v2** tables read through `Table.scan(...)` and its `to_arrow`,
  `to_arrow_batch_reader`, `to_pandas`, `to_polars`, `to_duckdb` and `count`
  outputs.
- Parquet **data** files and Parquet **equality delete** files on local storage
  (and any FileIO the table already uses).
- Delete keys over top-level `boolean`, `int`, `long`, `string` columns, single
  or composite, nullable.
- Coexistence with the existing positional (`position`) delete support.

Out of scope / rejected loudly:

- Delete-key types other than boolean/int/long/string raise `ValueError`
  (no stringification, no silent coercion).
- Non-Parquet equality delete files raise `ValueError`.
- Nested delete keys raise `ValueError` (only top-level keys are supported).

## Applicability (scan planning) — the standard rules

An equality delete file applies to a data file when (Iceberg v2 Scan Planning):

1. **data sequence number is strictly less** than the delete file's data
   sequence number — data added in the same or a later commit is never removed
   by that delete; and
2. either the partitions are equal (**same spec id** and equal partition values,
   with float equality by bit layout and NaN normalised), **or** the delete
   file's partition spec is **unpartitioned**, making it a **global** delete.

This logic lives in `pyiceberg/equality_deletes.py` and is the single source of
truth; `DataScan.plan_files` attaches the applicable delete files to each
`FileScanTask.equality_delete_files`. The reader never re-derives applicability.

Field identity is always by **field id**, never by column name, so renames are
safe.

## Execution design (two organisations compared)

### Option A — Shared key stores, probed by many data files (chosen)

Each distinct delete key is encoded once into a canonical, self-delimiting binary
token (type tag + null flag + payload). One *key store* per delete file holds the
token set; every data file the delete applies to probes that store with a
vectorised `is_in` over a batch-sized token column.

- High delete-key cardinality: memory tracks the number of *distinct* keys, not
  the number of delete rows; the set spills to temporary Arrow IPC segments when
  it exceeds the budget.
- Cross-data-file reuse: one store serves all applicable data files, so
  delete-file I/O and decode work do **not** grow with the number of data files.
- Parallelism: delete-file decode is serialised per file (each file once),
  probes are serialised per scan (one transient index at a time), while data-file
  reads stay parallel in the existing executor.
- Read amplification: each delete file is decoded **once per scan**.
- Temporary space: at most one IPC copy of the distinct tokens per delete file,
  in budget-sized segments.

### Option B — Materialise every delete file, then scan-join (rejected)

Read each delete file fully into an Arrow table and join it against every data
batch.

- Memory ∝ *total* delete rows (duplicates included), not distinct keys.
- The budget can only be checked after the file is already resident.
- Delete files are re-read (or cached in a second, unbounded layer) for every
  data file — read amplification grows with the number of data files.
- For the headline scenario (many data files sharing one large delete set) it is
  strictly worse on memory, I/O and budget enforcement.

Option A was chosen. Its cost is paid once per scan (one decode+encode pass,
plus one write+read of the token set when it spills), not once per data file.

### Streaming, non-materialising ingest

`EqualityDeleteContext._ensure_loaded` reads delete files with
`ParquetFile.iter_batches(...)`, encodes and inserts each batch, and checks the
budget per batch. An oversized single key is therefore reported after the first
batch that contains it — not after the whole file is materialised. (Verified:
with the huge key in the first row group, the error fires after 8 192 of 20 001
rows are read.)

### Spill and probe

When the resident key set would exceed the budget, the store moves **all**
resident tokens to temporary Arrow IPC **segments** (each sized to a fraction of
the budget) and releases their reservation. Probing reads one segment at a time.
If even a single segment's probe index does not fit the remaining budget, the
segment is probed in value-ordered sub-slices so a very small budget still makes
progress. A last-resort single-key path surfaces a clear `ValueError` only when
the budget cannot even hold one key.

## Memory budget

`Table.scan(options={"equality-delete-memory-budget": "<bytes>"})`:

- Value: decimal string of a positive integer, in bytes.
- Default `67108864` (64 MiB); minimum `65536` (64 KiB).
- Invalid values (`0`, negatives, `abc`, `65536.0`, `1e6`, below minimum, empty)
  raise `ValueError` **before any data page is opened** (validated in
  `plan_files`, which runs before the first file read).
- Per **scan**: one `BudgetAccount` shared by all parallel file tasks. Probes are
  serialised under a per-scan gate, so the transient probe index is never
  multiplied by concurrency.

What the budget covers vs. what it does not:

| Charged to the budget (equality-delete state only) | Accounted separately (never charged) |
|---|---|
| Retained token payloads | Regular data-read record batches |
| Per-key obj/container/index overheads (conservative) | The final output table |
| Probe hash index (per key, plus transient per segment) | Filesystem page cache |
| Per-batch match bitmap (4 bytes/row) | Fixed Parquet decoder overheads |
| Spill working buffer and pending buffer | |

Accounting is conservative: every Python `bytes` object, container entry, Arrow
offset buffer and index slot is charged a non-zero overhead, so large numbers of
tiny keys are not under-counted.

## Correctness properties exercised

Verified with real Parquet + real Avro manifests (see
`tests/io/test_equality_deletes.py`):

- Strict sequence-number rule (smaller / equal / larger delete sequence).
- Global (unpartitioned-spec) deletes hitting all partitions; partition-scoped
  deletes only hitting their own partition; differing spec ids not matching.
- Composite keys, null == null, duplicate delete keys, duplicate delete files.
- Renamed key, dropped key, key added after a data file (reads as null),
  int→long promotion, `equality_ids` order and subset.
- Empty delete file, all-rows-deleted, projecting only unrelated columns.
- Filters do not prune applicable deletes; `limit` applies after deletes/filter.
- `to_arrow`, `to_arrow_batch_reader` and unlimited `count` agree.
- Mixed position + equality deletes, with position deletes spanning row groups
  and original row numbers preserved under a filter.
- Unsupported key type rejected.

And (see `tests/io/test_equality_deletes_budget.py`):

- Two budgets (`65536`, `1048576`) give identical rows for the same snapshot.
- Delete set decoded volume ≫ the larger budget; small budget spills, large
  budget stays resident (or spills less); peak resident state stays within the
  budget.
- Spill files are removed on success, injected read failure, and early
  `to_arrow_batch_reader()` close.
- Source data/metadata/delete files are not modified.
- Interleaved scans at different budgets and two snapshots do not share results.

## Reproduced measurements

Hardware: local container, Python 3.11, PyArrow 19.0.1. Scenario: 4 data files ×
300 rows sharing one equality delete file with **260 000 distinct keys**
(delete file 1 358 314 bytes on disk; decoded delete volume 10 920 000 bytes).

| budget (bytes) | rows | delete file reads | delete bytes decoded | decoded / budget | peak resident | spill files | spill peak bytes | elapsed |
|---|---|---|---|---|---|---|---|---|
| 65 536 | 1200 | 1 | 10 920 000 | 166.6× | 6 656 | 1 | ≈14 200 000 | 0.69 s |
| 1 048 576 | 1200 | 1 | 10 920 000 | 10.4× | 35 328 | 1 | ≈13 400 000 | 0.45 s |

Key observations: identical row counts; each delete file decoded **once** even
though four data files share it; resident state stays within each budget
(6.6 KiB and 34.5 KiB); decoded volume is ≥8× the larger budget and ≥160× the
smaller. Spill-bytes are the total temporary IPC written and vary a little between
runs with segment framing, so they are shown as approximations.

## Reproducible commands

Environment setup used for these results (Python 3.11, PyArrow 19.0.1, PyIceberg
0.10.0 from this tree):

```bash
python3.11 -m venv /tmp/venv
/tmp/venv/bin/pip install pyarrow==19.0.1 numpy \
    "pydantic==2.10.6" "pyparsing==3.1.2" "pytest==7.4.4" "pytest-lazy-fixture==0.6.3" \
    mmh3 "requests<3" "click<9" "rich<15" "strictyaml<2" sortedcontainers \
    "fsspec>=2023.1.0" "zstandard<1" "tenacity<10" "pyroaring<2" "cachetools>=5.5,<7" \
    boto3 moto pytest-mock requests-mock fastavro "sqlalchemy>=2,<3" "pandas<3" \
    pyiceberg_core duckdb polars
```

The repository has a Cython Avro decoder that needs Python headers not present in
this sandbox, so the pure-Python decoder is used. Its one-time warning is
pre-existing and filtered on the command line.

New and affected tests:

```bash
/tmp/venv/bin/python -m pytest \
  tests/io/test_equality_deletes.py \
  tests/io/test_equality_deletes_budget.py \
  -q -p no:cacheprovider \
  -W "ignore:Falling back to pure Python Avro decoder:UserWarning"
# 48 passed

/tmp/venv/bin/python -m pytest \
  tests/io/test_equality_deletes.py tests/io/test_equality_deletes_budget.py \
  tests/table --ignore=tests/table/test_datafusion.py --ignore=tests/table/test_upsert.py \
  -q -p no:cacheprovider \
  -W "ignore:Falling back to pure Python Avro decoder:UserWarning"
# 264 passed

/tmp/venv/bin/python -m pytest tests/io/test_pyarrow.py \
  -q -p no:cacheprovider \
  -W "ignore:Falling back to pure Python Avro decoder:UserWarning" \
  -m "not gcs and not s3 and not adls and not hdfs and not aws" \
  -k "not gcs and not s3 and not adls and not hdfs"
# 115 passed, 62 deselected (cloud/network markers)
```

Pre-existing, environment-dependent failures deliberately excluded (confirmed by
running them on the unmodified tree):

- `tests/expressions/test_parser.py::test_quoted_column_with_dots` and
  `::test_invalid_between` — pyparsing version messages differ.
- `tests/table/test_datafusion.py`, `tests/table/test_upsert.py` — require the
  optional `datafusion` extra.

## Public usage

```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog("default")
table = catalog.load_table("db.events")

# Equality deletes are applied automatically.
rows = table.scan(row_filter="region = 'eu'").to_arrow()

# Bound the extra equality-delete memory for this scan (bytes, decimal string).
rows = table.scan(
    options={"equality-delete-memory-budget": "67108864"},
).to_arrow()

# Streaming reads release equality-delete state on close.
with_clause_free = table.scan().to_arrow_batch_reader()
try:
    for batch in with_clause_free:
        ...
finally:
    with_clause_free.close()
```
