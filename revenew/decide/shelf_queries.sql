-- Named SQL for revenew/decide/shelf.py, mirroring detect/queries.sql's
-- "-- name: <label>" convention (parsed the same way) but living in decide/,
-- not detect/, because this is not a detector -- it never produces an
-- opportunity, only a BUNDLE_OFFER ingredient for Arm B's template shelf.
--
-- name: global_bundle_pair
-- The single strongest product-pair affinity across the WHOLE order history,
-- with NO customer filter -- deliberately unlike detect/queries.sql's
-- cross_sell_affinity, which ranks one recommendation PER CUSTOMER. Arm B's
-- shelf must be cohort-level (see decide/shelf.py's module docstring): if
-- this query looked up one customer's own best pairing, Arm B would carry
-- information Arm C's LLM is never given (it only ever sees the merchant's
-- full catalog, never any one customer's basket -- decide/generator.py's
-- `_prompt_context`), which would make any A/B/C comparison unsound before a
-- single decision is made.
--
-- Same pair-count/confidence logic as cross_sell_affinity, just pooled
-- across every customer instead of partitioned by one.
WITH pair_counts AS (
    SELECT oi1.sku AS sku_from, oi2.sku AS sku_to, COUNT(DISTINCT oi1.order_id) AS pair_count
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.sku != oi2.sku
    JOIN orders o ON o.order_id = oi1.order_id
    WHERE o.status = 'captured'
    GROUP BY oi1.sku, oi2.sku
),
sku_order_counts AS (
    SELECT oi.sku, COUNT(DISTINCT oi.order_id) AS order_count
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.status = 'captured'
    GROUP BY oi.sku
),
strong_affinity AS (
    SELECT
        pc.sku_from, pc.sku_to,
        CAST(pc.pair_count AS REAL) / soc.order_count AS confidence
    FROM pair_counts pc
    JOIN sku_order_counts soc ON soc.sku = pc.sku_from
    WHERE pc.pair_count >= :min_pair_count
      AND CAST(pc.pair_count AS REAL) / soc.order_count >= :min_confidence
)
-- Ties broken by SKU code, not left to SQLite's unspecified row order, so
-- the same fixture always yields the same pair -- ShelfGenerator memoises
-- the result and a replay run must reproduce byte-identically.
SELECT sku_from, sku_to, confidence
FROM strong_affinity
ORDER BY confidence DESC, sku_from ASC, sku_to ASC
LIMIT 1;
