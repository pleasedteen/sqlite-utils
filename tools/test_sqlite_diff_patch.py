"""
Tests for sqlite_diff_patch.py — run with:

    cd tools && python3 -m unittest -v
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlite_diff_patch import (  # noqa: E402
    DiffError,
    generate_patch,
    quote_ident,
    sql_literal,
    topological_order,
    introspect,
)

SCHEMA = """
CREATE TABLE parent (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    note TEXT
);
CREATE TABLE child (
    a INTEGER NOT NULL,
    b INTEGER NOT NULL,
    parent_id INTEGER REFERENCES parent(id),
    label TEXT,
    PRIMARY KEY (a, b)
);
"""


def make_db(schema: str, *, fk_on: bool = True) -> tuple[str, sqlite3.Connection]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    if fk_on:
        conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(schema)
    return path, conn


def apply_patch(path: str, patch: str, *, pre_enable_fk: bool = True) -> None:
    """Apply a patch exactly as an external client with FK checks on would."""
    conn = sqlite3.connect(path)
    try:
        if pre_enable_fk:
            conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(patch)
        # FK violations must surface at COMMIT even if deferred.
        violations = list(conn.execute("PRAGMA foreign_key_check"))
        assert not violations, f"foreign key violations after patch: {violations}"
    finally:
        conn.close()


def dump(conn: sqlite3.Connection, table: str) -> list[tuple]:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM {table} "
        f"ORDER BY {', '.join(cols)}"
    ).fetchall()
    return rows


class RenderingTests(unittest.TestCase):
    def test_quote_ident(self) -> None:
        self.assertEqual(quote_ident('we"ird'), '"we""ird"')

    def test_sql_literal(self) -> None:
        self.assertEqual(sql_literal(None), "NULL")
        self.assertEqual(sql_literal(True), "1")
        self.assertEqual(sql_literal(False), "0")
        self.assertEqual(sql_literal(42), "42")
        self.assertEqual(sql_literal(1.5), "1.5")
        self.assertEqual(sql_literal("a'b"), "'a''b'")
        self.assertEqual(sql_literal(b"\x00\xff"), "X'00ff'")
        self.assertEqual(sql_literal("héllo 🎉"), "'héllo 🎉'")


class BasicDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.src_path, self.src = make_db(SCHEMA)
        self.tgt_path, self.tgt = make_db(SCHEMA)

    def tearDown(self) -> None:
        self.src.close()
        self.tgt.close()
        os.unlink(self.src_path)
        os.unlink(self.tgt_path)

    def test_insert_update_delete_single_pk(self) -> None:
        # Row 2 changes only the non-unique "note" column, so it must be
        # reconciled with an in-place UPDATE (the unique "name" stays put).
        self.src.executemany(
            "INSERT INTO parent (id, name, note) VALUES (?, ?, ?)",
            [(1, "keep", "same"), (2, "same-name", "new note"), (3, "only-src", None)],
        )
        self.tgt.executemany(
            "INSERT INTO parent (id, name, note) VALUES (?, ?, ?)",
            [(1, "keep", "same"), (2, "same-name", "old note"), (4, "only-tgt", "x")],
        )
        self.src.commit()
        self.tgt.commit()

        patch = generate_patch(self.src_path, self.tgt_path)
        self.assertIn("INSERT INTO", patch)
        self.assertIn("UPDATE ", patch)
        self.assertIn("DELETE FROM", patch)
        self.assertIn("PRAGMA defer_foreign_keys=ON", patch)

        apply_patch(self.tgt_path, patch)
        check = sqlite3.connect(self.tgt_path)
        self.assertEqual(dump(check, "parent"), dump(self.src, "parent"))
        check.close()

    def test_composite_primary_key_where_clause(self) -> None:
        self.src.execute(
            "INSERT INTO child (a, b, parent_id, label) VALUES (1, 9, NULL, 'x')"
        )
        self.tgt.execute(
            "INSERT INTO child (a, b, parent_id, label) VALUES (1, 9, NULL, 'y')"
        )
        self.src.commit()
        self.tgt.commit()
        patch = generate_patch(self.src_path, self.tgt_path)
        # Composite key must appear in full in the WHERE clause.
        self.assertIn('WHERE "a" = 1 AND "b" = 9;', patch)
        self.assertNotIn("DELETE FROM", patch)
        apply_patch(self.tgt_path, patch)
        check = sqlite3.connect(self.tgt_path)
        self.assertEqual(dump(check, "child"), dump(self.src, "child"))
        check.close()

    def test_null_in_composite_primary_key(self) -> None:
        # SQLite allows NULL inside a composite primary key (except
        # WITHOUT ROWID), so this needs a schema without NOT NULL columns.
        schema = "CREATE TABLE t (a INTEGER, b INTEGER, v TEXT, PRIMARY KEY (a, b));"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.execute("INSERT INTO t (a, b, v) VALUES (1, NULL, 'src')")
            tc.execute("INSERT INTO t (a, b, v) VALUES (1, NULL, 'tgt')")
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            self.assertIn('"b" IS NULL', patch)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute("SELECT a, b, v FROM t").fetchall(), [(1, None, "src")]
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_no_differences(self) -> None:
        self.src.execute("INSERT INTO parent (id, name) VALUES (1, 'a')")
        self.tgt.execute("INSERT INTO parent (id, name) VALUES (1, 'a')")
        self.src.commit()
        self.tgt.commit()
        patch = generate_patch(self.src_path, self.tgt_path)
        self.assertIn("No data differences found", patch)
        self.assertNotIn("INSERT", patch)

    def test_regenerated_patch_is_empty_after_apply(self) -> None:
        self.src.execute("INSERT INTO parent (id, name) VALUES (1, 'a')")
        self.src.commit()
        apply_patch(self.tgt_path, generate_patch(self.src_path, self.tgt_path))
        again = generate_patch(self.src_path, self.tgt_path)
        self.assertIn("No data differences found", again)

    def test_blob_and_special_text_roundtrip(self) -> None:
        self.src.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, b BLOB, s TEXT)"
        )
        self.tgt.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, b BLOB, s TEXT)"
        )
        self.src.execute(
            "INSERT INTO t VALUES (1, X'deadbeef', 'quote''s and 中文')"
        )
        self.src.commit()
        patch = generate_patch(self.src_path, self.tgt_path)
        apply_patch(self.tgt_path, patch)
        check = sqlite3.connect(self.tgt_path)
        row = check.execute("SELECT b, s FROM t").fetchone()
        self.assertEqual(row, (b"\xde\xad\xbe\xef", "quote's and 中文"))
        check.close()


class ForeignKeyTests(unittest.TestCase):
    def test_parent_child_ordering_and_cascade_delete(self) -> None:
        schema = """
        CREATE TABLE p (id INTEGER PRIMARY KEY, v TEXT);
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            p_id INTEGER NOT NULL REFERENCES p(id)
        );
        """
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            # Source: only parent 2 survives; target has parent 1 with a child.
            sc.execute("INSERT INTO p VALUES (2, 'new')")
            tc.executescript(
                "INSERT INTO p VALUES (1, 'old');"
                "INSERT INTO c VALUES (10, 1);"
            )
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            delete_pos = patch.index("Phase 1")
            insert_pos = patch.index("Phase 3")
            # Child delete must precede parent delete within phase 1.
            self.assertLess(
                patch.index("DELETE FROM \"c\"", delete_pos),
                patch.index("DELETE FROM \"p\"", delete_pos),
            )
            # Parent insert must precede any child insert in phase 3.
            self.assertLess(
                patch.index('INSERT INTO "p"', insert_pos),
                patch.index('INSERT INTO "c"', insert_pos)
                if 'INSERT INTO "c"' in patch[insert_pos:]
                else len(patch),
            )
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(cc.execute("SELECT id FROM p").fetchall(), [(2,)])
            self.assertEqual(cc.execute("SELECT id FROM c").fetchall(), [])
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_foreign_key_cycle_with_deferred_check(self) -> None:
        schema = """
        CREATE TABLE node (
            id INTEGER PRIMARY KEY,
            peer INTEGER REFERENCES node(id)
        );
        INSERT INTO node VALUES (1, NULL);
        INSERT INTO node VALUES (2, 1);
        """
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            # Source swaps the peer pointers: needs deferred enforcement.
            sc.executescript(
                "UPDATE node SET peer = 2 WHERE id = 1;"
                "UPDATE node SET peer = NULL WHERE id = 2;"
            )
            sc.commit()
            patch = generate_patch(sp, tp)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute("SELECT id, peer FROM node ORDER BY id").fetchall(),
                [(1, 2), (2, None)],
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_two_table_cycle(self) -> None:
        schema = """
        CREATE TABLE a (
            id INTEGER PRIMARY KEY,
            b_id INTEGER REFERENCES b(id)
        );
        CREATE TABLE b (
            id INTEGER PRIMARY KEY,
            a_id INTEGER REFERENCES a(id)
        );
        """
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            # Build the cross-reference without an immediate FK violation:
            # create a first with NULL, then b, then back-fill a.b_id.
            sc.executescript(
                "INSERT INTO a VALUES (1, NULL);"
                "INSERT INTO b VALUES (100, 1);"
                "UPDATE a SET b_id = 100 WHERE id = 1;"
            )
            sc.commit()
            patch = generate_patch(sp, tp)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(cc.execute("SELECT * FROM a").fetchall(), [(1, 100)])
            self.assertEqual(cc.execute("SELECT * FROM b").fetchall(), [(100, 1)])
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_parent_primary_key_change_rerouted(self) -> None:
        schema = """
        CREATE TABLE p (id INTEGER PRIMARY KEY, v TEXT);
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            p_id INTEGER NOT NULL REFERENCES p(id)
        );
        """
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            # Parent PK 1 -> 7; child re-pointed. Deferred FK makes it legal.
            sc.executescript(
                "INSERT INTO p VALUES (7, 'x');"
                "INSERT INTO c VALUES (10, 7);"
            )
            tc.executescript(
                "INSERT INTO p VALUES (1, 'x');"
                "INSERT INTO c VALUES (10, 1);"
            )
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            # PK change must be DELETE + INSERT, never an in-place UPDATE of id.
            self.assertNotIn('UPDATE "p" SET "id"', patch)
            self.assertIn('DELETE FROM "p" WHERE "id" = 1;', patch)
            self.assertIn('INSERT INTO "p"', patch)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(cc.execute("SELECT * FROM p").fetchall(), [(7, "x")])
            self.assertEqual(cc.execute("SELECT * FROM c").fetchall(), [(10, 7)])
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


    def test_failed_patch_rolls_back_entire_transaction(self) -> None:
        # The source itself contains an orphan child (built with FK off).
        # The deferred check must fail at COMMIT; an explicit ROLLBACK (the
        # standard SQLite recovery step, same as in the sqlite3 CLI) then
        # restores the target to its original state.
        schema = """
        CREATE TABLE p (id INTEGER PRIMARY KEY);
        CREATE TABLE c (
            id INTEGER PRIMARY KEY,
            p_id INTEGER NOT NULL REFERENCES p(id)
        );
        """
        sp, sc = make_db(schema, fk_on=False)
        tp, tc = make_db(schema)
        try:
            sc.execute("INSERT INTO c VALUES (1, 999)")  # parent 999 missing
            sc.commit()
            tc.execute("INSERT INTO p VALUES (42)")
            tc.commit()
            patch = generate_patch(sp, tp)
            conn = sqlite3.connect(tp)
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                conn.executescript(patch)
            except sqlite3.IntegrityError:
                # COMMIT failed: the deferred transaction is still open.
                self.assertTrue(conn.in_transaction)
                conn.rollback()
            else:  # pragma: no cover - failure is expected
                self.fail("patch should have failed the deferred FK check")
            conn.close()
            cc = sqlite3.connect(tp)
            # Target untouched: no orphan row, original data still present.
            self.assertEqual(cc.execute("SELECT * FROM p").fetchall(), [(42,)])
            self.assertEqual(cc.execute("SELECT * FROM c").fetchall(), [])
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


class UniqueConstraintTests(unittest.TestCase):
    def test_unique_value_swap_is_rerouted(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, code TEXT UNIQUE);"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.executescript(
                "INSERT INTO t VALUES (1, 'B');"
                "INSERT INTO t VALUES (2, 'A');"
            )
            tc.executescript(
                "INSERT INTO t VALUES (1, 'A');"
                "INSERT INTO t VALUES (2, 'B');"
            )
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            # In-place UPDATEs of a unique column would collide; both reroute.
            self.assertNotIn("UPDATE ", patch)
            self.assertEqual(patch.count("DELETE FROM"), 2)
            self.assertEqual(patch.count("INSERT INTO"), 2)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute("SELECT * FROM t ORDER BY id").fetchall(),
                [(1, "B"), (2, "A")],
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_non_unique_column_stays_inplace_update(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.execute("INSERT INTO t VALUES (1, 'new')")
            tc.execute("INSERT INTO t VALUES (1, 'old')")
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            self.assertIn("UPDATE ", patch)
            self.assertNotIn("DELETE FROM", patch)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(cc.execute("SELECT v FROM t").fetchall(), [("new",)])
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


class RowidAndWithoutRowidTests(unittest.TestCase):
    def test_rowid_table_no_declared_pk(self) -> None:
        schema = "CREATE TABLE t (name TEXT, age INTEGER);"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.executemany(
                "INSERT INTO t VALUES (?, ?)", [("a", 1), ("b", 2), ("c", 3)]
            )
            tc.executemany(
                "INSERT INTO t VALUES (?, ?)", [("a", 1), ("b", 99)]
            )
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            self.assertIn("rowid", patch)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute("SELECT name, age FROM t ORDER BY name").fetchall(),
                [("a", 1), ("b", 2), ("c", 3)],
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_without_rowid_composite_pk(self) -> None:
        schema = (
            "CREATE TABLE t (a INTEGER, b INTEGER, v TEXT, PRIMARY KEY (a, b)) "
            "WITHOUT ROWID;"
        )
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.executescript(
                "INSERT INTO t VALUES (1, 1, 'x');"
                "INSERT INTO t VALUES (2, 2, 'z');"
            )
            tc.execute("INSERT INTO t VALUES (1, 1, 'y')")
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute("SELECT * FROM t ORDER BY a").fetchall(),
                [(1, 1, "x"), (2, 2, "z")],
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


class MetaTests(unittest.TestCase):
    def test_schema_mismatch_raises(self) -> None:
        sp, sc = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT);")
        tp, tc = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY, b TEXT);")
        try:
            with self.assertRaises(DiffError):
                generate_patch(sp, tp)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_missing_table_raises(self) -> None:
        sp, sc = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY);")
        tp, tc = make_db("CREATE TABLE u (id INTEGER PRIMARY KEY);")
        try:
            with self.assertRaises(DiffError):
                generate_patch(sp, tp)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_views_are_skipped(self) -> None:
        schema = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
            "CREATE VIEW vw AS SELECT * FROM t;"
        )
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.execute("INSERT INTO t VALUES (1, 'a')")
            sc.commit()
            patch = generate_patch(sp, tp)
            self.assertIn("vw", patch)  # listed as skipped in the header
            apply_patch(tp, patch)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_deterministic_output(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.executemany(
                "INSERT INTO t VALUES (?, ?)",
                [(3, "c"), (1, "a"), (2, "b")],
            )
            sc.commit()
            p1 = generate_patch(sp, tp)
            p2 = generate_patch(sp, tp)
            self.assertEqual(p1, p2)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_topological_order(self) -> None:
        schema = """
        CREATE TABLE p (id INTEGER PRIMARY KEY);
        CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER REFERENCES p(id));
        CREATE TABLE g (id INTEGER PRIMARY KEY, c_id INTEGER REFERENCES c(id));
        """
        sp, conn = make_db(schema)
        try:
            tables, _ = introspect(conn)
            order = topological_order(tables)
            self.assertLess(order.index("p"), order.index("c"))
            self.assertLess(order.index("c"), order.index("g"))
        finally:
            conn.close()
            os.unlink(sp)


class IdentifierAndTypeTests(unittest.TestCase):
    def test_keyword_table_and_column_names(self) -> None:
        # "order" and "select" are SQL keywords; quoting must make them legal.
        schema = (
            'CREATE TABLE "order" ('
            '"select" INTEGER PRIMARY KEY, "group" TEXT NOT NULL UNIQUE);'
        )
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.executescript(
                'INSERT INTO "order" VALUES (1, \'a\');'
                'INSERT INTO "order" VALUES (2, \'b\');'
            )
            tc.executescript(
                'INSERT INTO "order" VALUES (1, \'a\');'
                'INSERT INTO "order" VALUES (3, \'c\');'
            )
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            apply_patch(tp, patch)
            cc = sqlite3.connect(tp)
            self.assertEqual(
                cc.execute('SELECT * FROM "order" ORDER BY 1').fetchall(),
                [(1, "a"), (2, "b")],
            )
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_numeric_storage_class_equality(self) -> None:
        # 1 (integer) and 1.0 (real) compare equal in Python, mirroring
        # SQLite's value equality; no churn is emitted for such pairs.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, v);"
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            sc.execute("INSERT INTO t VALUES (1, 1)")
            tc.execute("INSERT INTO t VALUES (1, 1.0)")
            sc.commit()
            tc.commit()
            patch = generate_patch(sp, tp)
            self.assertIn("No data differences found", patch)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)

    def test_cli_exits_nonzero_on_schema_mismatch(self) -> None:
        sp, sc = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT);")
        tp, tc = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY, b TEXT);")
        try:
            result = subprocess.run(
                [sys.executable, "-I",
                 str(Path(__file__).with_name("sqlite_diff_patch.py")),
                 sp, tp],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


class RandomizedFuzzTests(unittest.TestCase):
    def test_random_mutations_converge(self) -> None:
        import random

        schema = """
        CREATE TABLE p (id INTEGER PRIMARY KEY, kind TEXT);
        CREATE TABLE c (
            a INTEGER, b INTEGER,
            p_id INTEGER REFERENCES p(id),
            tag TEXT,
            PRIMARY KEY (a, b)
        );
        """
        rng = random.Random(20261112)
        sp, sc = make_db(schema)
        tp, tc = make_db(schema)
        try:
            kinds = ["alpha", "beta", "gamma", "delta"]
            for db in (sc, tc):
                db.executemany(
                    "INSERT INTO p (id, kind) VALUES (?, ?)",
                    [(i, k) for i, k in enumerate(kinds)],
                )
                db.executemany(
                    "INSERT INTO c (a, b, p_id, tag) VALUES (?, ?, ?, ?)",
                    [
                        (i, j, rng.randrange(len(kinds)), f"t{i}-{j}")
                        for i in range(5)
                        for j in range(5)
                    ],
                )
                db.commit()

            # Mutate the source independently; keep every FK value valid.
            def mutate(db: sqlite3.Connection) -> None:
                for _ in range(40):
                    table = rng.choice(["p", "c"])
                    if table == "p":
                        op = rng.choice(["update", "noop", "noop"])
                        if op == "update":
                            row = db.execute(
                                "SELECT id FROM p ORDER BY RANDOM() LIMIT 1"
                            ).fetchone()
                            if row:
                                db.execute(
                                    "UPDATE p SET kind = ? WHERE id = ?",
                                    (rng.choice(kinds) + str(rng.randrange(1000)),
                                     row[0]),
                                )
                    else:
                        op = rng.choice(["insert", "delete", "update"])
                        if op == "insert":
                            a, b = rng.randrange(10, 99), rng.randrange(10, 99)
                            exists = db.execute(
                                "SELECT 1 FROM c WHERE a=? AND b=?", (a, b)
                            ).fetchone()
                            if not exists:
                                db.execute(
                                    "INSERT INTO c VALUES (?, ?, ?, ?)",
                                    (a, b, rng.randrange(4), f"n{a}"),
                                )
                        elif op == "delete":
                            row = db.execute(
                                "SELECT a, b FROM c ORDER BY RANDOM() LIMIT 1"
                            ).fetchone()
                            if row:
                                db.execute(
                                    "DELETE FROM c WHERE a=? AND b=?", row
                                )
                        else:
                            row = db.execute(
                                "SELECT a, b FROM c ORDER BY RANDOM() LIMIT 1"
                            ).fetchone()
                            if row:
                                db.execute(
                                    "UPDATE c SET tag=?, p_id=? WHERE a=? AND b=?",
                                    (f"u{rng.randrange(10000)}",
                                     rng.randrange(4), *row),
                                )
                db.commit()

            mutate(sc)
            mutate(tc)
            patch = generate_patch(sp, tp)
            apply_patch(tp, patch)

            cc = sqlite3.connect(tp)
            for table in ("p", "c"):
                self.assertEqual(dump(cc, table), dump(sc, table))
            cc.close()
        finally:
            sc.close()
            tc.close()
            os.unlink(sp)
            os.unlink(tp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
