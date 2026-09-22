"""SQLite 数据库：连接、建表、灌入示例数据、schema 内省。

示例数据是一份电商业务数据集（客户 / 商品 / 订单 / 订单明细），
足够支撑「趋势 / 占比 / 排名 / 对比」等典型分析问题。
"""
from __future__ import annotations

import datetime
import random
import sqlite3
from pathlib import Path
from typing import List, Optional

from .config import DB_PATH

# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 建表与示例数据
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    city        TEXT,
    segment     TEXT,                -- '个人' / '企业'
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT,
    price      REAL,
    cost       REAL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  TEXT,
    status      TEXT,                -- '已完成' / '已取消' / '待支付'
    channel     TEXT                 -- '线上' / '门店' / '分销'
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER,
    unit_price    REAL
);
"""

# (名称, 品类, 售价, 成本)
PRODUCTS = [
    ("iPhone 15", "电子产品", 6999, 5200),
    ("MacBook Air", "电子产品", 8999, 7200),
    ("AirPods Pro", "电子产品", 1899, 1200),
    ("索尼降噪耳机", "电子产品", 2299, 1500),
    ("北欧实木餐桌", "家居", 2599, 1600),
    ("记忆棉床垫", "家居", 3999, 2400),
    ("落地灯", "家居", 499, 260),
    ("运动跑鞋", "运动", 899, 450),
    ("瑜伽垫", "运动", 199, 90),
    ("羽绒服", "服饰", 1299, 700),
    ("牛仔裤", "服饰", 399, 180),
    ("精装图书套装", "图书", 599, 300),
]

CUSTOMER_NAMES = [
    "张伟", "王芳", "李娜", "刘洋", "陈静", "杨磊", "赵敏", "黄强",
    "周杰", "吴霞", "徐涛", "孙丽", "马超", "朱琳", "胡军", "郭燕",
    "何平", "高翔", "林峰", "罗丹", "郑浩", "梁雪", "谢宇", "唐婷",
]

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]


def init_db(db_path: Path | str = DB_PATH) -> None:
    """建表并灌入示例数据（幂等：已有数据则跳过）。"""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        if _is_empty(conn):
            _seed(conn)
        conn.commit()
    finally:
        conn.close()


def _is_empty(conn: sqlite3.Connection) -> bool:
    cur = conn.execute("SELECT COUNT(*) FROM customers")
    return cur.fetchone()[0] == 0


def _seed(conn: sqlite3.Connection) -> None:
    rng = random.Random(42)

    customers = []
    for i, name in enumerate(CUSTOMER_NAMES, start=1):
        signup = datetime.date(2023, rng.randint(1, 6), rng.randint(1, 28)).isoformat()
        customers.append(
            (i, name, f"user{i}@example.com", rng.choice(CITIES),
             rng.choice(["个人", "企业"]), signup)
        )
    conn.executemany(
        "INSERT INTO customers (customer_id, name, email, city, segment, signup_date) "
        "VALUES (?,?,?,?,?,?)", customers
    )
    conn.executemany(
        "INSERT INTO products (product_id, name, category, price, cost) VALUES (?,?,?,?,?)",
        [(i + 1, *p) for i, p in enumerate(PRODUCTS)],
    )

    start = datetime.date(2023, 1, 1)
    end = datetime.date(2024, 12, 31)
    days = (end - start).days

    orders: List[tuple] = []
    order_items: List[tuple] = []
    oid = 1
    iid = 1
    for cid in range(1, len(CUSTOMER_NAMES) + 1):
        for _ in range(rng.randint(2, 6)):
            d = start + datetime.timedelta(days=rng.randint(0, days))
            status = rng.choices(["已完成", "已取消", "待支付"], weights=[80, 8, 12])[0]
            channel = rng.choice(["线上", "门店", "分销"])
            orders.append((oid, cid, d.isoformat(), status, channel))
            for _ in range(rng.randint(1, 3)):
                pid = rng.randint(1, len(PRODUCTS))
                qty = rng.randint(1, 5)
                up = round(PRODUCTS[pid - 1][2] * rng.uniform(0.9, 1.1), 2)
                order_items.append((iid, oid, pid, qty, up))
                iid += 1
            oid += 1

    conn.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, status, channel) "
        "VALUES (?,?,?,?,?)", orders
    )
    conn.executemany(
        "INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) "
        "VALUES (?,?,?,?,?)", order_items
    )


# ---------------------------------------------------------------------------
# schema 内省
# ---------------------------------------------------------------------------

def list_tables(db_path: Path | str = DB_PATH) -> List[str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_table_schema(table: str, conn: Optional[sqlite3.Connection] = None) -> str:
    """返回单张表的结构信息：DDL + 列说明 + 行数 + 3 行样例数据。"""
    own = conn is None
    if own:
        conn = get_connection()
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        create_sql = row[0] if row else ""
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT 1')
        columns = [d[0] for d in cur.description] if cur.description else []
        samples = [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()]

        lines = [f"## 表: {table}", "", "### 列", ", ".join(columns) or "(无)", ""]
        if create_sql:
            lines += ["### DDL", "```sql", create_sql, "```", ""]
        lines += ["### 行数", str(cnt), "", "### 示例数据", "```json"]
        for s in samples:
            lines.append(_json_compat(s))
        lines += ["```"]
        return "\n".join(lines)
    finally:
        if own:
            conn.close()


def get_relevant_schema(question: str, db_path: Path | str = DB_PATH) -> str:
    """根据问题关键词，挑选相关的表；匹配不到时返回全部表结构。

    关键词表可根据业务扩展。对于当前 4 张表的示例数据，通常直接返回全部。
    """
    tables = list_tables(db_path)
    # 关键词 -> 相关表 映射（用于大库场景下的粗粒度检索）
    keyword_map = {
        "订单": ["orders", "order_items"],
        "销售": ["orders", "order_items", "products"],
        "客户": ["customers"],
        "产品": ["products"],
        "商品": ["products"],
        "品类": ["products"],
        "城市": ["customers"],
    }
    matched = []
    for kw, tbls in keyword_map.items():
        if kw in question:
            matched.extend(tbls)
    relevant = list(dict.fromkeys(matched))  # 去重保序

    # 若关键词没有命中任何表，或命中了过多，直接返回全部表结构
    if not relevant:
        relevant = tables

    conn = get_connection(db_path)
    try:
        parts = [get_table_schema(t, conn) for t in relevant if t in tables]
    finally:
        conn.close()
    return "\n\n".join(parts)


def _json_compat(d: dict) -> str:
    """把样例行渲染成紧凑的 JSON 字符串（确保中文可读）。"""
    import json
    return json.dumps(d, ensure_ascii=False)
