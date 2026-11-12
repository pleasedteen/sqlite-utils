#!/usr/bin/env python3
"""
sqlite_diff_patch.py
====================

Compare the table *data* of two SQLite databases that share the same schema
and emit a replayable SQL patch that turns the *target* database into an exact
data-level copy of the *source* database.

Features
--------

* INSERT rows present only in the source, DELETE rows present only in the
  target, UPDATE rows whose columns differ.
* Correct handling of composite primary keys (all PK columns are used in the
  WHERE clauses) and of rowid tables that have no declared primary key.
* Foreign-key safe ordering:

  - deletes run children-before-parents, inserts run parents-before-children
    (topological sort over the foreign-key graph, cycles included);
  - the whole patch is one transaction with
    ``PRAGMA defer_foreign_keys=ON`` so that parent primary-key changes and
    foreign-key column updates never raise mid-patch, even with
    ``PRAGMA foreign_keys=ON`` active on the target connection.
* Rows whose primary key or any UNIQUE-indexed column changes are applied as
  DELETE + INSERT (in separate global phases) so that swapping unique values
  between two rows cannot trigger a transient UNIQUE constraint failure.
* Deterministic output (rows are emitted in primary-key / rowid order); no
  timestamps, so generated patches are diff-friendly.

Limitations
-----------

* Schemas must match (same tables, columns and declared column types). Only
  data is patched; schema migration is out of scope.
* Virtual tables (e.g. FTS), shadow tables (``sqlite_*``) and views are
  skipped; they are listed in a header comment.
* Triggers are not disabled: if the target has triggers that enforce extra
  rules or cascade writes, those still fire while the patch is applied.
* If the deferred foreign-key check fails at ``COMMIT`` (for example because
  the source database itself contains an invalid reference), SQLite keeps the
  transaction open rather than auto-rolling-back; issue ``ROLLBACK;`` to
  restore the target, exactly as you would with the ``sqlite3`` CLI.

Usage
-----

    python3 sqlite_diff_patch.py source.db target.db -o patch.sql
    sqlite3 target.db < patch.sql

Library usage::

    from sqlite_diff_patch import generate_patch
    sql = generate_patch("source.db", "target.db")
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DiffError",
    "ForeignKey",
    "TableDiff",
    "TableInfo",
    "generate_patch",
    "quote_ident",
    "sql_literal",
]


class DiffError(RuntimeError):
    """Raised when the two databases cannot be compared (e.g. schema drift)."""


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str | None  # NULL means "referenced table's primary key"


@dataclass
class TableInfo:
    name: str
    columns: list[str]
    column_types: list[str]
    pk_columns: list[str]
    # Each tuple is the column set of a UNIQUE index (including the implicit
    # primary-key index). A change touching any of these columns is rerouted
    # through DELETE + INSERT to avoid transient unique collisions.
    unique_groups: list[tuple[str, ...]]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    without_rowid: bool = False

    @property
    def key_columns(self) -> list[str]:
        """Columns that uniquely identify a row, or ``['rowid']``."""
        return list(self.pk_columns) if self.pk_columns else ["rowid"]


@dataclass
class TableDiff:
    table: str
    key_columns: list[str]
    inserts: list[dict] = field(default_factory=list)
    # (old key tuple, {column: new value})
    updates: list[tuple[tuple, dict]] = field(default_factory=list)
    deletes: list[tuple] = field(default_factory=list)
    # (old key tuple, full new row dict) for updates rerouted via DELETE+INSERT
    rerouted: list[tuple[tuple, dict]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.inserts or self.updates or self.deletes or self.rerouted)


# --------------------------------------------------------------------------- #
# SQL rendering
# --------------------------------------------------------------------------- #

def quote_ident(name: str) -> str:
    """Quote a SQLite identifier (table/column name)."""
    return '"' + name.replace('"', '""') + '"'


def sql_literal(value: object) -> str:
    """Render a Python value (as returned by sqlite3) as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Inf/NaN cannot be stored by SQLite anyway; emit NULL defensively.
        if value != value or value in (float("inf"), float("-inf")):
            return "NULL"
        return repr(value)
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _where_key(key_columns: list[str], key_values: tuple) -> str:
    clauses = []
    for col, val in zip(key_columns, key_values):
        if val is None:
            clauses.append(f"{quote_ident(col)} IS NULL")
        else:
            clauses.append(f"{quote_ident(col)} = {sql_literal(val)}")
    return " AND ".join(clauses)


def _insert_statement(table: str, row: dict, columns: list[str]) -> str:
    col_sql = ", ".join(quote_ident(c) for c in columns)
    val_sql = ", ".join(sql_literal(row[c]) for c in columns)
    return f"INSERT INTO {quote_ident(table)} ({col_sql}) VALUES ({val_sql});"


def _update_statement(
    table: str, key_columns: list[str], key_values: tuple, changes: dict
) -> str:
    set_sql = ", ".join(
        f"{quote_ident(c)} = {sql_literal(v)}" for c, v in changes.items()
    )
    return (
        f"UPDATE {quote_ident(table)} SET {set_sql} "
        f"WHERE {_where_key(key_columns, key_values)};"
    )


def _delete_statement(
    table: str, key_columns: list[str], key_values: tuple
) -> str:
    where = _where_key(key_columns, key_values)
    return f"DELETE FROM {quote_ident(table)} WHERE {where};"


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #

def _pragma(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    return list(conn.execute(sql))


def introspect(conn: sqlite3.Connection) -> tuple[dict[str, TableInfo], list[str]]:
    """Return (user tables by name, names of skipped objects)."""
    tables: dict[str, TableInfo] = {}
    skipped: list[str] = []

    master = _pragma(
        conn,
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE type IN ('table', 'view') ORDER BY name",
    )
    for name, obj_type, sql_text in master:
        if name.startswith("sqlite_"):
            skipped.append(name)
            continue
        if obj_type == "view":
            skipped.append(name)
            continue
        if sql_text is not None and sql_text.lstrip().upper().startswith(
            "CREATE VIRTUAL TABLE"
        ):
            skipped.append(name)
            continue

        info_rows = _pragma(conn, f"PRAGMA table_info({quote_ident(name)})")
        columns = [r[1] for r in info_rows]
        column_types = [r[2] for r in info_rows]
        # table_info.pk is 1-based position in the primary key, 0 = not a PK.
        pk_rows = sorted(
            ((r[5], r[1]) for r in info_rows if r[5] > 0), key=lambda r: r[0]
        )
        pk_columns = [r[1] for r in pk_rows]

        unique_groups: list[tuple[str, ...]] = []
        for idx in _pragma(conn, f"PRAGMA index_list({quote_ident(name)})"):
            # Columns vary by SQLite version; positional access is stable:
            # (seq, index_name, unique, origin, ...)
            index_name, is_unique = idx[1], idx[2]
            if not is_unique:
                continue
            idx_info = _pragma(
                conn, f"PRAGMA index_info({quote_ident(index_name)})"
            )
            group = tuple(r[2] for r in sorted(idx_info, key=lambda r: r[0]))
            if group:
                unique_groups.append(group)

        # INTEGER PRIMARY KEY is a rowid alias and has no entry in
        # index_list, and composite-PK rowid tables only get an implicit
        # index: register the primary key explicitly as a unique group so
        # changes to key columns are always rerouted through DELETE+INSERT.
        if pk_columns:
            unique_groups.append(tuple(pk_columns))
        # Deduplicate while keeping first-seen order.
        unique_groups = list(dict.fromkeys(unique_groups))

        fks: list[ForeignKey] = []
        for fk in _pragma(conn, f"PRAGMA foreign_key_list({quote_ident(name)})"):
            # (id, seq, ref_table, from_col, to_col, ...)
            fks.append(ForeignKey(column=fk[3], ref_table=fk[2], ref_column=fk[4]))

        without_rowid = bool(
            sql_text and "WITHOUT ROWID" in sql_text.upper()
        )
        tables[name] = TableInfo(
            name=name,
            columns=columns,
            column_types=column_types,
            pk_columns=pk_columns,
            unique_groups=unique_groups,
            foreign_keys=fks,
            without_rowid=without_rowid,
        )

    return tables, skipped


def _check_compatible(
    source_tables: dict[str, TableInfo], target_tables: dict[str, TableInfo]
) -> None:
    problems: list[str] = []
    only_source = sorted(set(source_tables) - set(target_tables))
    only_target = sorted(set(target_tables) - set(source_tables))
    if only_source:
        problems.append(f"tables only in source: {', '.join(only_source)}")
    if only_target:
        problems.append(f"tables only in target: {', '.join(only_target)}")
    for name in sorted(set(source_tables) & set(target_tables)):
        src, tgt = source_tables[name], target_tables[name]
        if src.columns != tgt.columns:
            problems.append(
                f"table {name!r}: column mismatch "
                f"(source {src.columns} vs target {tgt.columns})"
            )
        elif src.column_types != tgt.column_types:
            problems.append(
                f"table {name!r}: declared column type mismatch "
                f"(source {src.column_types} vs target {tgt.column_types})"
            )
        if src.pk_columns != tgt.pk_columns:
            problems.append(f"table {name!r}: primary key mismatch")
    if problems:
        raise DiffError(
            "databases do not share a comparable schema:\n  - "
            + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------- #
# Row comparison
# --------------------------------------------------------------------------- #

def _fetch_rows(
    conn: sqlite3.Connection, info: TableInfo
) -> list[tuple[tuple, dict]]:
    """Return [(identity key, row dict), ...] ordered deterministically."""
    table = quote_ident(info.name)
    if info.pk_columns:
        names = list(info.columns)
        sql = (
            f"SELECT {', '.join(quote_ident(c) for c in names)} "
            f"FROM {table} ORDER BY "
            f"{', '.join(quote_ident(c) for c in info.pk_columns)}"
        )
        rows = [dict(zip(names, raw)) for raw in conn.execute(sql)]
        return [
            (tuple(row[c] for c in info.pk_columns), row) for row in rows
        ]

    names = ["__diff_rowid__", *info.columns]
    sql = (
        f"SELECT rowid AS __diff_rowid__, "
        f"{', '.join(quote_ident(c) for c in info.columns)} "
        f"FROM {table} ORDER BY rowid"
    )
    out: list[tuple[tuple, dict]] = []
    for raw in conn.execute(sql):
        record = dict(zip(names, raw))
        key = (record["__diff_rowid__"],)
        out.append((key, {c: record[c] for c in info.columns}))
    return out


def _needs_reroute(info: TableInfo, changed_columns: set[str]) -> bool:
    """True if an in-place UPDATE could hit a UNIQUE (incl. PK) constraint."""
    for group in info.unique_groups:
        if changed_columns.intersection(group):
            return True
    return False


def diff_table(
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    info: TableInfo,
) -> TableDiff:
    # Single full scan per side; ordering inside these lists is what makes
    # the emitted statement order deterministic.
    source_scan = _fetch_rows(source_conn, info)
    target_scan = _fetch_rows(target_conn, info)
    source_rows = dict(source_scan)
    target_rows = dict(target_scan)

    result = TableDiff(table=info.name, key_columns=info.key_columns)

    # Rows only in the source: INSERT.
    for key, row in source_scan:
        if key not in target_rows:
            result.inserts.append(row)

    # Rows only in the target: DELETE; shared rows: compare column by column.
    for key, target_row in target_scan:
        if key not in source_rows:
            result.deletes.append(key)
            continue
        source_row = source_rows[key]
        changes = {
            col: source_row[col]
            for col in info.columns
            if source_row[col] != target_row[col]
        }
        if not changes:
            continue
        if _needs_reroute(info, set(changes)):
            result.rerouted.append((key, source_row))
        else:
            result.updates.append((key, changes))

    return result


# --------------------------------------------------------------------------- #
# Ordering and patch rendering
# --------------------------------------------------------------------------- #

def topological_order(tables: dict[str, TableInfo]) -> list[str]:
    """Parents before children, based on foreign keys; cycles are broken.

    Iterative DFS so long dependency chains and cycles cannot exhaust the
    recursion limit. Back edges (cycles) are ignored: the deferred-FK
    transaction makes cyclic inserts/deletes legal anyway.
    """
    parents: dict[str, set[str]] = {
        name: {
            fk.ref_table
            for fk in info.foreign_keys
            if fk.ref_table in tables and fk.ref_table != name
        }
        for name, info in tables.items()
    }
    order: list[str] = []
    done: set[str] = set()

    for root in sorted(tables):
        if root in done:
            continue
        # Stack entries are (table, iterator over that table's parents).
        stack: list[tuple[str, Iterator[str]]] = [
            (root, iter(sorted(parents[root])))
        ]
        on_stack: set[str] = {root}
        while stack:
            name, it = stack[-1]
            advanced = False
            for parent in it:
                if parent in done or parent in on_stack:
                    continue  # already emitted, or a back edge of a cycle
                stack.append((parent, iter(sorted(parents[parent]))))
                on_stack.add(parent)
                advanced = True
                break
            if not advanced:
                stack.pop()
                on_stack.discard(name)
                done.add(name)
                order.append(name)
    return order


def generate_patch(
    source: str | Path | sqlite3.Connection,
    target: str | Path | sqlite3.Connection,
) -> str:
    """Build the SQL patch that makes *target*'s data equal *source*'s."""
    own_conns: list[sqlite3.Connection] = []

    def connect(db: str | Path | sqlite3.Connection) -> sqlite3.Connection:
        if isinstance(db, sqlite3.Connection):
            return db
        conn = sqlite3.connect(str(db))
        own_conns.append(conn)
        return conn

    source_conn = connect(source)
    target_conn = connect(target)
    try:
        source_tables, source_skipped = introspect(source_conn)
        target_tables, target_skipped = introspect(target_conn)
        _check_compatible(source_tables, target_tables)

        order = topological_order(target_tables)
        diffs = {
            name: diff_table(source_conn, target_conn, target_tables[name])
            for name in order
        }

        lines: list[str] = [
            "-- Data reconciliation patch generated by sqlite_diff_patch.py",
            f"-- Source database : {_label(source)}",
            f"-- Target database : {_label(target)}",
            "-- Direction       : transform target so its data matches source",
        ]
        skipped = sorted(set(source_skipped) | set(target_skipped))
        if skipped:
            lines.append(
                "-- Skipped objects (views / virtual / internal): "
                + ", ".join(skipped)
            )

        has_changes = any(not d.is_empty() for d in diffs.values())
        if not has_changes:
            lines.append("-- No data differences found; nothing to do.")
            return "\n".join(lines) + "\n"

        lines.extend(
            [
                "",
                "-- Must be set outside a transaction; no-op if already on.",
                "PRAGMA foreign_keys=ON;",
                "BEGIN;",
                "-- Delay all FK enforcement until COMMIT, so parent key",
                "-- changes and cyclic references never fail mid-patch.",
                "PRAGMA defer_foreign_keys=ON;",
            ]
        )

        # Phase 1: deletes, children first (reverse of parent-first order).
        # Rerouted updates delete their old row here as well.
        lines.append("")
        lines.append("-- Phase 1: deletes (children before parents)")
        for name in reversed(order):
            diff = diffs[name]
            info = target_tables[name]
            for key in diff.deletes:
                lines.append(_delete_statement(name, diff.key_columns, key))
            for old_key, _ in diff.rerouted:
                lines.append(_delete_statement(name, diff.key_columns, old_key))

        # Phase 2: ordinary in-place updates (no unique-index columns touched).
        lines.append("")
        lines.append("-- Phase 2: in-place updates")
        emitted_update = False
        for name in order:
            diff = diffs[name]
            for key, changes in diff.updates:
                lines.append(
                    _update_statement(name, diff.key_columns, key, changes)
                )
                emitted_update = True
        if not emitted_update:
            lines.append("-- (none)")

        # Phase 3: inserts, parents first. Rerouted rows re-enter here.
        lines.append("")
        lines.append("-- Phase 3: inserts (parents before children)")
        for name in order:
            diff = diffs[name]
            info = target_tables[name]
            for row in diff.inserts:
                lines.append(_insert_statement(name, row, info.columns))
            for _, row in diff.rerouted:
                lines.append(_insert_statement(name, row, info.columns))

        lines.extend(["COMMIT;", ""])
        return "\n".join(lines)
    finally:
        for conn in own_conns:
            conn.close()


def _label(db: str | Path | sqlite3.Connection) -> str:
    if isinstance(db, sqlite3.Connection):
        return f"connection {db!r}"
    return str(db)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare data of two identically-structured SQLite databases and "
            "emit a replayable SQL patch that transforms TARGET into SOURCE."
        )
    )
    parser.add_argument("source", help="Database whose data is the desired end state")
    parser.add_argument("target", help="Database to be patched")
    parser.add_argument(
        "-o",
        "--output",
        help="Write the SQL patch here (default: stdout)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        patch = generate_patch(args.source, args.target)
    except DiffError as exc:
        raise SystemExit(f"error: {exc}") from None
    if args.output:
        Path(args.output).write_text(patch, encoding="utf-8")
    else:
        print(patch, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
