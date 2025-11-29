import logging
from typing import Optional, Dict

import pandas as pd
import psycopg2
from psycopg2.extras import Json

from .config import DATABASE_URL

logger = logging.getLogger(__name__)


def get_connection():
    if not DATABASE_URL:
        logger.info("DATABASE_URL не задан — пропускаю запись в БД")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        return conn
    except Exception as exc:
        logger.warning("Не удалось подключиться к БД: %s", exc)
        return None


def is_available() -> bool:
    conn = get_connection()
    if not conn:
        return False
    try:
        conn.close()
        return True
    except Exception:
        return False


def insert_batch(
    conn,
    *,
    file_name: str,
    purpose: str,
    status: str,
    total_rows: int,
    class_counts: Dict[str, int],
    macro_f1: Optional[float] = None,
    f1_per_class: Optional[Dict[str, float]] = None,
    support: Optional[Dict[str, int]] = None,
    output_path: Optional[str] = None,
):
    sql = """
    INSERT INTO batches (file_name, purpose, status, total_rows, class_counts, macro_f1, f1_per_class, support, output_path)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                file_name,
                purpose,
                status,
                total_rows,
                Json(class_counts),
                macro_f1,
                Json(f1_per_class) if f1_per_class is not None else None,
                Json(support) if support is not None else None,
                output_path,
            ),
        )
        batch_id = cur.fetchone()[0]
    return batch_id


def insert_reviews(conn, *, batch_id, df, max_rows: int):
    if max_rows <= 0:
        return 0
    subset = df.head(max_rows)
    rows = []
    for idx, row in subset.iterrows():
        rows.append(
            (
                int(idx),
                row["text"],
                int(row["label"]) if "label" in row and not pd.isna(row["label"]) else None,
                int(row["predicted_label"]),
                row.get("predicted_scores"),
            )
        )
    sql = """
    INSERT INTO reviews (row_idx, input_text, true_label, predicted_label, predicted_scores, batch_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                sql,
                (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    Json(row[4]) if row[4] else None,
                    batch_id,
                ),
            )
    return len(rows)
