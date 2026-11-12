# sqlite_diff_patch

零第三方依赖（仅标准库 `sqlite3`）的 SQLite 数据差异补丁工具。对比两个**结构相同**的
SQLite 库的表数据，生成可直接重放的 SQL 补丁，使目标库（target）的数据与源库（source）
完全一致，支持新增 / 修改 / 删除三类变更。

## 用法

```bash
# 生成补丁（默认输出到 stdout，-o 写入文件）
python3 -I sqlite_diff_patch.py source.db target.db -o patch.sql

# 在开启外键检查的连接上重放
sqlite3 target.db < patch.sql
```

库形式调用：

```python
from sqlite_diff_patch import generate_patch

sql = generate_patch("source.db", "target.db")
```

## 关键设计

1. **三阶段单事务**：`PRAGMA foreign_keys=ON` + `BEGIN` +
   `PRAGMA defer_foreign_keys=ON`，删除（子→父）→ 原地更新 → 插入（父→子），
   最后 `COMMIT` 时统一做外键检查。父表主键变更、外键列变更、外键环在补丁执行
   中途都不会报错。
2. **复合主键**：WHERE 子句包含全部主键列，主键中的 NULL 用 `IS NULL` 匹配；
   无显式主键的 rowid 表用 `rowid` 作为行身份；支持 `WITHOUT ROWID` 表。
3. **唯一约束安全**：任何触及主键或 UNIQUE 索引列的修改都会改走
   DELETE + INSERT（在两个全局阶段中完成），因此两行互换唯一值不会产生
   瞬时唯一冲突；只改普通列时保留原地 UPDATE。
4. **确定性输出**：按主键 / rowid 排序输出，无时间戳，补丁本身可 diff。
5. **结构校验**：表、列、列声明类型、主键不一致时直接报错，不生成补丁。
   视图、虚拟表（如 FTS）、`sqlite_*` 内部表跳过并在文件头注释中列出。

注意：若源库自身存在非法外键，延迟检查会在 `COMMIT` 处失败，此时事务仍处于打开
状态（SQLite 标准行为），执行 `ROLLBACK;` 即可完整还原目标库。

## 测试

```bash
python3 -I -m unittest -v
```

26 个用例覆盖：复合主键 / NULL 主键、rowid 与 WITHOUT ROWID 表、外键环（自引用与
两表互引）、父表主键改值、唯一值互换、BLOB/Unicode/关键字标识符、事务回滚、
schema 漂移报错以及随机变更模糊测试（打补丁后两库数据必须逐表相等）。
