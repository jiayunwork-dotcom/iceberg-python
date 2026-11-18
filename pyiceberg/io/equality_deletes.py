# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Bounded-memory execution of Iceberg v2 equality deletes.

Scan planning decides *which* equality delete files apply to a data file (see
``DataScan.plan_files``).  This module only executes that decision; it never
re-derives applicability from data sequence numbers, spec ids or partitions.

Design
------
Two organisations of the work were evaluated:

1. **Shared key sets, probed by many data files (chosen).**  Each distinct delete
   key is encoded once into a canonical, self-delimiting byte token (type
   tagged, null flagged).  One *key store* exists per delete-key signature
   (``equality_ids`` + canonical key types) and is shared read-only by every
   data file whose applicable deletes have that signature.  A data batch is
   probed with a vectorised ``is_in`` against the store.  Resident memory is
   proportional to the number of *distinct* delete keys, not to the number of
   delete rows or to the number of data files; delete files are streamed in
   row-group batches so an oversized file is detected before it is fully
   materialised; and each delete file is decoded once per scan.

2. **Materialise every delete file, then scan-join.**  Read each delete file
   fully into an Arrow table and join it against every data batch.  Simpler, but
   resident memory is proportional to the *total* delete rows (duplicates
   included), the budget can only be checked after the fact, and delete files
   must be re-read (or cached in a second unbounded layer) for every data file.

The chosen organisation pays one decode+encode pass over the distinct delete
keys per scan and, when a single key store is too large to stay resident, an
extra write+read of that store as temporary Arrow IPC files.  It does not pay
delete-file I/O per data file.

Memory accounting
-----------------
``equality-delete-memory-budget`` bounds *only* the state equality delete
matching adds on top of a normal read: retained key payloads, the match/probe
index, per-store metadata and spill buffers.  Regular data-read batches, the
final output, the filesystem page cache and fixed Parquet decoder overheads are
accounted separately and never charged here.

Accounting is conservative: every Python object, container entry and index slot
is charged a non-zero overhead, so a large number of tiny keys is never
under-counted.  A single distinct key that cannot fit into the whole budget
raises :class:`EqualityDeleteMemoryError` (a ``ValueError``) as soon as it is
decoded, before the rest of that delete file is read.
"""

from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from pyiceberg.equality_deletes import EqualityDeleteFile
from pyiceberg.exceptions import ResolveError
from pyiceberg.io import FileIO

# ---------------------------------------------------------------------------
# Conservative accounting constants
# ---------------------------------------------------------------------------
# CPython 3.11 ``bytes.__sizeof__`` is 33 + len.  A retained token is also a set
# entry (hash slot + pointer) and has an entry in the offsets array.
PY_BYTES_OVERHEAD = 33
PY_POINTER = 8
CONTAINER_ENTRY_OVERHEAD = 64
ARROW_OFFSET_BYTES = 8
# The probe index (hash table used by ``is_in``) is counted per key store and
# shared by all data files.  64 bytes per key covers a ~2x-slot table, the key
# pointer and allocator slack, and is deliberately generous.
MATCH_INDEX_BYTES_PER_KEY = 64
# Per key store: the Python object, lock, bounds and bookkeeping.
KEY_STORE_OVERHEAD = 512
# Per probe: the match bitmap is 1 byte per row; count 4 to stay conservative
# (Arrow bool buffer, numpy view, allocator rounding).
MATCH_MASK_BYTES_PER_ROW = 4
# Streamed decode batch size for delete files.
DELETE_DECODE_BATCH_SIZE = 8192

_TOKEN_TYPE_INT = b"\x01"
_TOKEN_TYPE_BOOL = b"\x02"
_TOKEN_TYPE_STRING = b"\x03"
_TOKEN_NULL = b"\x00"
_EMPTY = pa.scalar(b"", pa.binary())

BOOLEAN = "boolean"
LONG = "long"
STRING = "string"

_FIELD_ID_METADATA_KEY = b"PARQUET:field_id"


class EqualityDeleteMemoryError(ValueError):
    """Raised when equality delete state cannot fit into the configured budget."""


# ---------------------------------------------------------------------------
# Delete-key signature
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KeySpec:
    """The delete-key signature: ordered field ids and canonical key types."""

    field_ids: Tuple[int, ...]
    types: Tuple[str, ...]

    @staticmethod
    def from_delete_file(delete_file: EqualityDeleteFile) -> "KeySpec":
        return KeySpec(tuple(delete_file.equality_ids), tuple(delete_file.key_types))


def _canonical_arrow_type(type_name: str) -> pa.DataType:
    if type_name == LONG:
        return pa.int64()
    if type_name == BOOLEAN:
        return pa.bool_()
    if type_name == STRING:
        return pa.string()
    raise ValueError(f"Unsupported equality key type {type_name}")


# ---------------------------------------------------------------------------
# Token encoding: canonical, type-tagged, null-aware byte strings
# ---------------------------------------------------------------------------
# Component token layout (self-delimiting, so concatenation is unambiguous):
#   0x00                                       -> null
#   0x01 + 8 bytes                             -> integer (int and long)
#   0x02 + one byte (0/1)                      -> boolean
#   0x03 + 8 bytes length + UTF-8 payload      -> string
# Composite keys concatenate their components.  "All components equal" is then
# exactly "composite token equal", with null == null by construction, and
# distinct keys can never collide across types or column boundaries.
def _int64_as_8_bytes(values: pa.Array) -> pa.Array:
    """Reinterpret an int64 array's data buffer as 8-byte binary, zero copy.

    Only equality is required of the encoding, so the machine byte order is fine
    as long as it is used consistently for delete keys and data rows.
    """
    buffers = values.buffers()
    fixed = pa.Array.from_buffers(pa.binary(8), len(values), [buffers[0], buffers[1]], offset=values.offset)
    return pc.cast(fixed, pa.binary())


def _encode_integer(values: pa.Array) -> pa.Array:
    casted = pc.cast(values, pa.int64())
    payload = _int64_as_8_bytes(casted)
    tagged = pc.binary_join_element_wise(pa.scalar(_TOKEN_TYPE_INT, pa.binary()), payload, _EMPTY)
    return pc.if_else(casted.is_null(), pa.scalar(_TOKEN_NULL, pa.binary()), tagged)


def _encode_boolean(values: pa.Array) -> pa.Array:
    true_token = _TOKEN_TYPE_BOOL + b"\x01"
    false_token = _TOKEN_TYPE_BOOL + b"\x00"
    return pc.if_else(
        values.is_null(),
        pa.scalar(_TOKEN_NULL, pa.binary()),
        pc.if_else(values, pa.scalar(true_token, pa.binary()), pa.scalar(false_token, pa.binary())),
    )


def _encode_string(values: pa.Array) -> pa.Array:
    payload = pc.cast(values, pa.binary())
    lengths = pc.cast(pc.binary_length(payload), pa.int64())
    tagged = pc.binary_join_element_wise(pa.scalar(_TOKEN_TYPE_STRING, pa.binary()), _int64_as_8_bytes(lengths), payload, _EMPTY)
    return pc.if_else(values.is_null(), pa.scalar(_TOKEN_NULL, pa.binary()), tagged)


def _encode_component(values: pa.Array, type_name: str) -> pa.Array:
    if isinstance(values.type, pa.DictionaryType):
        values = values.dictionary_decode()
    if type_name == LONG:
        return _encode_integer(values)
    if type_name == BOOLEAN:
        return _encode_boolean(values)
    if type_name == STRING:
        return _encode_string(values)
    raise ValueError(f"Unsupported equality key type {type_name}")


def encode_key(batch: pa.RecordBatch, spec: KeySpec) -> pa.Array:
    """Encode a batch's key columns (in ``spec.field_ids`` order) into one token column."""
    components = [_encode_component(batch.column(position), type_name) for position, type_name in enumerate(spec.types)]
    if len(components) == 1:
        return components[0]
    return pc.binary_join_element_wise(*components, _EMPTY)


# ---------------------------------------------------------------------------
# Key-only batch construction from a data batch
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedEqualityDelete:
    """An equality delete resolved against one data file's physical schema.

    Attributes:
        spec: The canonical key signature (ordered field ids + types).
        columns: For each key field id, the physical column name in the data
            file, or ``None`` when the field is absent from that file (the key is
            then read as null, per Iceberg projection rules).
        file_path: The equality delete file path (diagnostics / dedup).
    """

    spec: KeySpec
    columns: Dict[int, Optional[str]]
    file_path: str


def build_key_batch(batch: pa.RecordBatch, descriptor: ResolvedEqualityDelete) -> pa.RecordBatch:
    """Build a key-only batch in ``spec.field_ids`` order."""
    arrays: List[pa.Array] = []
    for field_id, type_name in zip(descriptor.spec.field_ids, descriptor.spec.types):
        target_type = _canonical_arrow_type(type_name)
        column_name = descriptor.columns.get(field_id)
        if column_name is None:
            arrays.append(pa.nulls(len(batch), type=target_type))
            continue
        values = batch.column(column_name)
        if isinstance(values.type, pa.DictionaryType):
            values = values.dictionary_decode()
        if values.type != target_type:
            values = pc.cast(values, target_type)
        arrays.append(values)
    return pa.record_batch(arrays, names=[f"k{index}" for index in range(len(arrays))])


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------
class BudgetAccount:
    """A conservative, thread-safe byte account shared by all tasks in a scan."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.budget - self._used

    def reserve(self, amount: int, what: str = "equality delete state") -> None:
        with self._lock:
            if self._used + amount > self.budget:
                raise EqualityDeleteMemoryError(
                    f"Equality delete memory budget of {self.budget} bytes is exhausted while retaining {what}: "
                    f"{self._used} bytes in use, {amount} more requested"
                )
            self._used += amount

    def try_reserve(self, amount: int) -> bool:
        with self._lock:
            if self._used + amount > self.budget:
                return False
            self._used += amount
            return True

    def release(self, amount: int) -> None:
        with self._lock:
            self._used = max(0, self._used - amount)


def _token_resident_cost(lengths: Sequence[int]) -> int:
    """Conservative resident + probe-index bytes for distinct token payloads."""
    return sum(
        PY_BYTES_OVERHEAD
        + CONTAINER_ENTRY_OVERHEAD
        + PY_POINTER
        + ARROW_OFFSET_BYTES
        + MATCH_INDEX_BYTES_PER_KEY
        + max(length, 0)
        for length in lengths
    )


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------
# A spilled key set is a sequence of self-contained Arrow IPC segments.  Probing
# only ever keeps one segment (plus its probe index) resident, so the whole key
# set never has to fit.  Segment size is a fraction of the scan budget, chosen so
# that reading and probing one segment always fits alongside the pending buffer
# and the match bitmap.
_MAX_SEGMENT_TARGET_BYTES = 32768
_MIN_SEGMENT_TARGET_BYTES = 1024
# Conservative size of one in-flight probe segment: payload + offsets + validity
# + the Arrow array object + the probe hash table.
_SEGMENT_FIXED_COST = 2048


def _segment_target_bytes(budget: int) -> int:
    return max(_MIN_SEGMENT_TARGET_BYTES, min(_MAX_SEGMENT_TARGET_BYTES, budget // 16))


class KeyStore:
    """A shared token set for one delete-key signature.

    Built once per scan and shared read-only by every data file whose applicable
    equality deletes share the signature.  It lives in memory while it fits the
    scan budget and spills to temporary Arrow IPC segments when it does not.
    Spilled segments are probed one at a time, so the amount of key state that
    must be resident during a probe is bounded by a single segment rather than by
    the whole delete set.
    """

    def __init__(self, spec: KeySpec, account: BudgetAccount) -> None:
        self.spec = spec
        self._account = account
        self._lock = threading.RLock()
        self._closed = False
        self._chunks: List[pa.Array] = []
        self._spill_paths: List[str] = []
        self._resident_bytes = 0
        self._resident_accounted = False
        self._spill_accounted = 0
        self._spill_bytes = 0
        # Pending tokens not yet flushed into a segment.  The pending buffer is
        # charged to the same account when the first batch spills.
        self._pending: List[pa.Array] = []
        self._pending_bytes = 0
        self._segment_target = _segment_target_bytes(account.budget)
        self._pending_accounted = self._segment_target + _SEGMENT_FIXED_COST
        account.reserve(KEY_STORE_OVERHEAD, "equality delete key store header")
        self._store_accounted = KEY_STORE_OVERHEAD

    @property
    def spilled(self) -> bool:
        return len(self._spill_paths) > 0

    @property
    def spill_bytes(self) -> int:
        return self._spill_bytes

    # -- ingest ----------------------------------------------------------
    def add(self, tokens: pa.Array) -> None:
        if self._closed:
            raise RuntimeError("Cannot add to a closed equality delete key store")
        if len(tokens) == 0:
            return

        # Deduplicate within the batch: the budget is about distinct keys.  Global
        # deduplication is not required for correctness (a probe is a set
        # membership test) and would force materialising the whole store.
        unique = pc.unique(tokens)
        lengths = pc.binary_length(unique).to_pylist()

        # A single key that can never fit must be reported explicitly, before the
        # rest of the delete file is decoded.
        largest = max(lengths) if lengths else 0
        single_cost = _token_resident_cost([largest])
        if single_cost > self._account.budget:
            raise EqualityDeleteMemoryError(
                f"A single equality delete key for fields {list(self.spec.field_ids)} needs {single_cost} bytes, "
                f"which exceeds the entire equality-delete-memory-budget of {self._account.budget} bytes"
            )

        cost = _token_resident_cost(lengths)
        with self._lock:
            if self._spill_accounted == 0 and self._account.try_reserve(cost):
                self._chunks.append(unique)
                self._resident_bytes += cost
                self._resident_accounted = True
                return

        # Does not fit resident (or the store has begun spilling): spill.  Move any
        # keys already held resident to disk too, so the freed budget is available
        # for the match bitmap and the transient probe index.
        self._spill(unique)

    def _spill(self, tokens: pa.Array) -> None:
        with self._lock:
            if self._spill_accounted == 0:
                # Release resident keys first so the spill working buffer can be
                # charged even when the resident keys filled the budget.
                chunks, self._chunks = self._chunks, []
                resident_bytes, resident_accounted = self._resident_bytes, self._resident_accounted
                self._resident_bytes = 0
                self._resident_accounted = False
                if resident_accounted and resident_bytes:
                    self._account.release(resident_bytes)
                if not self._account.try_reserve(self._pending_accounted):
                    raise EqualityDeleteMemoryError(
                        f"Equality delete memory budget of {self._account.budget} bytes cannot hold the spill "
                        "working buffer; the key set does not fit resident and cannot be spilled"
                    )
                self._spill_accounted = self._pending_accounted
                if chunks:
                    resident = pc.unique(pa.chunked_array(chunks, type=pa.binary()).combine_chunks())
                    self._write_segment_split(resident)
                    del chunks, resident

            self._pending.append(tokens)
            self._pending_bytes += int(sum(pc.binary_length(tokens).to_pylist()))
            should_flush = self._pending_bytes >= self._segment_target
            if should_flush:
                self._flush_pending_locked()

    def flush_pending(self) -> None:
        with self._lock:
            self._flush_pending_locked()

    def _flush_pending_locked(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        self._pending_bytes = 0
        combined = pc.unique(pa.chunked_array(pending, type=pa.binary()).combine_chunks())
        self._write_segment_split(combined)

    def _write_segment_split(self, tokens: pa.Array) -> None:
        """Write ``tokens`` as one or more segments of at most the segment target."""
        lengths = pc.binary_length(tokens).to_pylist()
        start = 0
        total = len(tokens)
        while start < total:
            payload = 0
            end = start
            while end < total and (payload < self._segment_target or end == start):
                payload += max(lengths[end], 0)
                end += 1
            self._write_segment(tokens.slice(start, end - start))
            start = end

    def _write_segment(self, tokens: pa.Array) -> None:
        handle, path = tempfile.mkstemp(prefix="pyiceberg-eqdel-", suffix=".arrows")
        os.close(handle)
        try:
            with open(path, "wb") as out:
                with ipc.new_stream(out, pa.schema([pa.field("t", pa.binary())])) as writer:
                    writer.write_batch(pa.record_batch([tokens], names=["t"]))
                self._spill_bytes += out.tell()
        except BaseException:
            _safe_unlink(path)
            raise
        with self._lock:
            self._spill_paths.append(path)

    def finalize(self) -> None:
        """Flush any pending spilled tokens into a final segment."""
        self.flush_pending()

    def force_spill(self) -> None:
        """Move any resident keys to disk, freeing budget for probing / match masks.

        Safe to call at any time: a store holding only spilled segments is a no-op,
        and a probe over the resident + spilled sets already sees every key.
        """
        with self._lock:
            if self._closed or self._spill_paths or not self._chunks:
                return
            chunks, self._chunks = self._chunks, []
            resident_bytes, resident_accounted = self._resident_bytes, self._resident_accounted
            self._resident_bytes = 0
            self._resident_accounted = False
            # Release the resident reservation first: the budget may be entirely
            # consumed by these keys, leaving no room to charge the spill buffer.
            if resident_accounted and resident_bytes:
                self._account.release(resident_bytes)
            if self._spill_accounted == 0:
                if not self._account.try_reserve(self._pending_accounted):
                    # Cannot hold the spill working buffer; put the keys back.
                    self._chunks = chunks + self._chunks
                    self._resident_bytes += resident_bytes
                    self._resident_accounted = resident_accounted
                    return
                self._spill_accounted = self._pending_accounted
            resident = pc.unique(pa.chunked_array(chunks, type=pa.binary()).combine_chunks())
            self._write_segment_split(resident)

    # -- probing ---------------------------------------------------------
    def _read_segment(self, path: str) -> pa.Array:
        with open(path, "rb") as handle:
            reader = ipc.open_stream(handle)
            batches = [batch.column("t") for batch in reader]
            del reader
        if not batches:
            return pa.array([], type=pa.binary())
        return pa.chunked_array(batches, type=pa.binary()).combine_chunks()

    def _segment_probe_cost(self, values: pa.Array) -> int:
        return _token_resident_cost(pc.binary_length(values).to_pylist()) + _SEGMENT_FIXED_COST

    def probe(self, tokens: pa.Array) -> pa.Array:
        """Return a boolean array: True where the token exists in the store."""
        matched: Optional[pa.Array] = None
        for chunk in list(self._chunks):
            current = pc.is_in(tokens, value_set=chunk)
            matched = current if matched is None else pc.or_(matched, current)

        # Tokens that have not been flushed yet are still only in memory.
        if self._pending:
            with self._lock:
                pending = list(self._pending)
            for chunk in pending:
                current = pc.is_in(tokens, value_set=chunk)
                matched = current if matched is None else pc.or_(matched, current)

        for path in list(self._spill_paths):
            values = self._read_segment(path)
            segment_cost = self._segment_probe_cost(values)
            # The transient probe index must fit; if it does not, the segment is
            # narrowed by splitting it into smaller ranges below.
            if not self._account.try_reserve(segment_cost):
                current = self._probe_segment_narrowed(tokens, values)
            else:
                try:
                    current = pc.is_in(tokens, value_set=values)
                finally:
                    self._account.release(segment_cost)
            matched = current if matched is None else pc.or_(matched, current)

        if matched is None:
            return pa.array([False] * len(tokens), type=pa.bool_())
        return matched

    def _probe_segment_narrowed(self, tokens: pa.Array, values: pa.Array) -> pa.Array:
        """Probe a segment in value-order sub-ranges small enough for the budget.

        The probe-side tokens are never split or re-read: the *delete* segment is
        partitioned into slices, each of which is a fraction of the whole.  Every
        slice is probed against all data tokens and the results are OR-ed, so the
        surviving set is identical to a single ``is_in`` over the whole segment.
        Slice size is derived from the remaining budget so a tiny budget still
        makes progress (with a single-key fallback only if even that is too big).
        """
        total = len(values)
        if total == 0:
            return pa.array([False] * len(tokens), type=pa.bool_())

        # Approximate per-key cost, then choose the largest slice count that fits.
        payload = int(sum(pc.binary_length(values).to_pylist()))
        per_key = _token_resident_cost([payload // total if total else 0])
        available = max(self._account.remaining - _SEGMENT_FIXED_COST, 0)
        slice_size = max(1, available // max(per_key, 1))
        if slice_size >= total:
            slice_size = total

        matched: Optional[pa.Array] = None
        for start in range(0, total, slice_size):
            part = values.slice(start, min(slice_size, total - start))
            part_cost = self._segment_probe_cost(part)
            if not self._account.try_reserve(part_cost):
                current = self._probe_segment_single(tokens, part)
            else:
                try:
                    current = pc.is_in(tokens, value_set=part)
                finally:
                    self._account.release(part_cost)
            matched = current if matched is None else pc.or_(matched, current)
        assert matched is not None
        return matched

    def _probe_segment_single(self, tokens: pa.Array, part: pa.Array) -> pa.Array:
        """Last-resort probe comparing one key at a time (bounded resident state)."""
        matched = pa.array([False] * len(tokens), type=pa.bool_())
        for index in range(len(part)):
            single = part.slice(index, 1)
            single_cost = self._segment_probe_cost(single)
            if not self._account.try_reserve(single_cost):
                raise EqualityDeleteMemoryError(
                    f"Equality delete memory budget of {self._account.budget} bytes is too small to probe "
                    "a single delete key; increase equality-delete-memory-budget"
                )
            try:
                matched = pc.or_(matched, pc.is_in(tokens, value_set=single))
            finally:
                self._account.release(single_cost)
        return matched

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._chunks = []
            self._pending = []
            if self._resident_accounted and self._resident_bytes:
                self._account.release(self._resident_bytes)
            self._resident_bytes = 0
            self._resident_accounted = False
            if self._store_accounted:
                self._account.release(self._store_accounted)
                self._store_accounted = 0
            if self._spill_accounted:
                self._account.release(self._spill_accounted)
                self._spill_accounted = 0
            paths, self._spill_paths = self._spill_paths, []
        for path in paths:
            _safe_unlink(path)


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        # Cleanup must never mask a read result or an in-flight exception.
        pass


# ---------------------------------------------------------------------------
# Observable counters
# ---------------------------------------------------------------------------
@dataclass
class EqualityDeleteStats:
    """Counters describing one scan's equality delete execution."""

    delete_file_reads: int = 0
    delete_bytes_read: int = 0
    peak_resident_bytes: int = 0
    spill_files: int = 0
    spill_peak_bytes: int = 0
    key_stores: int = 0
    data_batches_filtered: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "delete_file_reads": self.delete_file_reads,
            "delete_bytes_read": self.delete_bytes_read,
            "peak_resident_bytes": self.peak_resident_bytes,
            "spill_files": self.spill_files,
            "spill_peak_bytes": self.spill_peak_bytes,
            "key_stores": self.key_stores,
            "data_batches_filtered": self.data_batches_filtered,
        }


# ---------------------------------------------------------------------------
# Per-scan context
# ---------------------------------------------------------------------------
class EqualityDeleteContext:
    """Scan-scoped equality delete state: one budget, shared key stores.

    A context is created per ``to_arrow`` / ``to_arrow_batch_reader`` / ``count``
    call and is never shared between snapshots or concurrent scans.
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.account = BudgetAccount(budget)
        # One store per delete file: a store is shared read-only by every data
        # file the delete file applies to, but its keys are never pooled with a
        # *different* delete file's keys (that would apply deletes to data files
        # the delete file does not govern).
        self._stores: Dict[str, KeyStore] = {}
        self._loaded: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._stats = EqualityDeleteStats()
        self._stats_lock = threading.Lock()
        # Probing the shared (possibly spilled) key stores needs a transient probe
        # index drawn from the same scan budget.  Serialising probes keeps that
        # transient cost to one probe at a time per scan, so the budget bound is
        # independent of how many data-file tasks run in parallel.
        self._probe_gate = threading.Lock()

    @property
    def stats(self) -> EqualityDeleteStats:
        return self._stats

    def _note_peak(self) -> None:
        used = self.account.used
        with self._stats_lock:
            if used > self._stats.peak_resident_bytes:
                self._stats.peak_resident_bytes = used
            spill = sum(store.spill_bytes for store in self._stores.values())
            if spill > self._stats.spill_peak_bytes:
                self._stats.spill_peak_bytes = spill
            self._stats.spill_files = sum(1 for store in self._stores.values() if store.spilled)

    def store_for(self, file_path: str, spec: KeySpec) -> KeyStore:
        with self._lock:
            store = self._stores.get(file_path)
            if store is None:
                store = KeyStore(spec, self.account)
                self._stores[file_path] = store
                with self._stats_lock:
                    self._stats.key_stores += 1
        self._note_peak()
        return store

    # -- resolution ------------------------------------------------------
    def resolve(
        self,
        delete_files: Sequence[EqualityDeleteFile],
        file_columns_by_field_id: Dict[int, str],
    ) -> List[ResolvedEqualityDelete]:
        """Resolve applicable delete files against a data file's physical columns."""
        resolved: List[ResolvedEqualityDelete] = []
        for delete_file in delete_files:
            spec = KeySpec.from_delete_file(delete_file)
            columns = {field_id: file_columns_by_field_id.get(field_id) for field_id in spec.field_ids}
            resolved.append(ResolvedEqualityDelete(spec=spec, columns=columns, file_path=delete_file.file_path))
        return resolved

    # -- loading ---------------------------------------------------------
    def prepare(self, io: FileIO, resolved: Sequence[ResolvedEqualityDelete]) -> None:
        """Load key stores for ``resolved`` (each delete file at most once per scan)."""
        for descriptor in resolved:
            self._ensure_loaded(io, descriptor.file_path, descriptor.spec)

    def _ensure_loaded(self, io: FileIO, file_path: str, spec: KeySpec) -> None:
        with self._lock:
            event = self._loaded.get(file_path)
            if event is None:
                event = threading.Event()
                self._loaded[file_path] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait()
            return

        store = self.store_for(file_path, spec)
        bytes_read = 0
        try:
            with io.new_input(file_path).open() as stream:
                parquet_file = pq.ParquetFile(stream)
                for batch in parquet_file.iter_batches(batch_size=DELETE_DECODE_BATCH_SIZE):
                    bytes_read += _batch_nbytes(batch)
                    store.add(encode_key(_project_delete_batch(batch, spec), spec))
                store.finalize()
        except BaseException:
            with self._lock:
                self._loaded.pop(file_path, None)
            event.set()
            raise
        with self._stats_lock:
            self._stats.delete_file_reads += 1
            self._stats.delete_bytes_read += bytes_read
        event.set()
        self._note_peak()

    # -- filtering -------------------------------------------------------
    def filter_batch(
        self,
        batch: pa.RecordBatch,
        descriptors: Sequence[ResolvedEqualityDelete],
    ) -> pa.RecordBatch:
        """Remove rows matched by any applicable equality delete.

        Returns a batch with the same columns as ``batch`` (the caller drops the
        auxiliary key columns afterwards).  Large data batches are processed in
        slices so the match bitmap it adds stays within the budget.
        """
        if not descriptors or batch.num_rows == 0:
            return batch

        with self._stats_lock:
            self._stats.data_batches_filtered += 1

        rows = batch.num_rows
        # The mask reservation and the probe index are both drawn from the shared
        # budget.  Serialising the whole filter step per scan bounds them to one
        # batch at a time, regardless of how many file tasks run in parallel.
        with self._probe_gate:
            mask_bytes = rows * MATCH_MASK_BYTES_PER_ROW
            if not self.account.try_reserve(mask_bytes):
                self._spill_resident_stores()
            if self.account.try_reserve(mask_bytes):
                try:
                    return self._filter_batch_rows(batch, descriptors)
                finally:
                    self.account.release(mask_bytes)

            # A batch larger than what remains: slice into budget-sized pieces.
            remaining = self.account.remaining
            rows_per_slice = max(1, remaining // MATCH_MASK_BYTES_PER_ROW)
            pieces: List[pa.RecordBatch] = []
            for start in range(0, rows, rows_per_slice):
                part = batch.slice(start, min(rows_per_slice, rows - start))
                slice_mask = part.num_rows * MATCH_MASK_BYTES_PER_ROW
                self.account.reserve(slice_mask, "equality delete match bitmap")
                try:
                    pieces.append(self._filter_batch_rows(part, descriptors))
                finally:
                    self.account.release(slice_mask)
            if not pieces:
                return batch.slice(0, 0)
            return pa.concat_batches(pieces)

    def _spill_resident_stores(self) -> None:
        """Force any store still holding keys in memory to spill to disk."""
        with self._lock:
            stores = list(self._stores.values())
        for store in stores:
            store.force_spill()

    def _filter_batch_rows(
        self,
        batch: pa.RecordBatch,
        descriptors: Sequence[ResolvedEqualityDelete],
    ) -> pa.RecordBatch:
        # The caller holds ``self._probe_gate``: one probe at a time per scan so
        # the transient index for a spilled store is charged exactly once.
        keep: Optional[pa.Array] = None
        for descriptor in descriptors:
            store = self._stores[descriptor.file_path]
            matched = store.probe(encode_key(build_key_batch(batch, descriptor), descriptor.spec))
            keep = pc.invert(matched) if keep is None else pc.and_(keep, pc.invert(matched))
        assert keep is not None
        if not pc.any(keep).as_py():
            return batch.slice(0, 0)
        return batch.filter(keep)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stores = list(self._stores.values())
        for store in stores:
            store.close()
        self._note_peak()

    def __enter__(self) -> "EqualityDeleteContext":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _batch_nbytes(batch: pa.RecordBatch) -> int:
    return sum(column.nbytes for column in batch.columns)


def _project_delete_batch(batch: pa.RecordBatch, spec: KeySpec) -> pa.RecordBatch:
    """Select delete-key columns by field id (order follows ``spec.field_ids``)."""
    id_to_position: Dict[int, int] = {}
    for position, field in enumerate(batch.schema):
        raw_id = field.metadata.get(_FIELD_ID_METADATA_KEY) if field.metadata else None
        if raw_id is None:
            continue
        try:
            id_to_position[int(raw_id)] = position
        except (TypeError, ValueError):
            continue
    arrays: List[pa.Array] = []
    for field_id in spec.field_ids:
        column_position = id_to_position.get(field_id)
        if column_position is None:
            raise ResolveError(f"Equality delete file is missing delete key field id {field_id}")
        arrays.append(batch.column(column_position))
    return pa.record_batch(arrays, names=[f"k{index}" for index in range(len(arrays))])


__all__ = [
    "BudgetAccount",
    "EqualityDeleteContext",
    "EqualityDeleteMemoryError",
    "EqualityDeleteStats",
    "KeySpec",
    "KeyStore",
    "ResolvedEqualityDelete",
    "build_key_batch",
    "encode_key",
]
