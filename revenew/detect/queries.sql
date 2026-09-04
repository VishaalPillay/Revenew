-- Named SQL, one block per OpportunityType. detector.py parses this file by
-- splitting on `-- name: <opportunity_type>` markers and hashes each block's
-- text verbatim -- that hash is what `detector_query_hash` records, so any
-- edit to a query here changes the hash every future-detected opportunity of
-- that type carries. This is the mechanism behind F1's "carries the query
-- that produced it": the query is not described, it is fingerprinted.
--
-- Every block is parameterised with :now, :dormant_threshold_days,
-- :first_order_threshold_days, :min_pair_count, :min_confidence -- detector.py
-- supplies all of them, none are literals here, so the same query text runs
-- identically under the wall clock and under a virtual clock mid-replay.
--
-- Each block must return exactly: customer_id, rupees_at_risk. detector.py
-- adds opportunity_id, opportunity_type, window_id, cohort_id, segment,
-- detector_query_hash, and detected_at itself -- none of that is SQL's job.

-- name: dormant_winback
-- A customer with at least one captured order and no order in
-- :dormant_threshold_days or more. Rupees at risk is their own historical
-- average order value: the plainest estimate of what one more order from THEM
-- specifically is worth, not a population average.
WITH customer_stats AS (
    SELECT
        c.customer_id,
        COUNT(o.order_id) AS orders_count,
        MAX(o.placed_at) AS last_order_at,
        AVG(o.amount) AS avg_order_value,
        CAST(julianday(:now) - julianday(MAX(o.placed_at)) AS INTEGER) AS days_since_last_order
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'captured'
    GROUP BY c.customer_id
)
SELECT customer_id, avg_order_value AS rupees_at_risk
FROM customer_stats
WHERE orders_count >= 1
  AND days_since_last_order >= :dormant_threshold_days;

-- name: first_order_retention
-- Exactly one captured order, placed at least :first_order_threshold_days
-- ago, with no second order since. Rupees at risk cannot be the customer's
-- own history -- they only have one data point -- so it is the average FIRST
-- reorder amount among customers who did come back, a real cohort estimate
-- rather than a guess.
WITH customer_stats AS (
    SELECT
        c.customer_id,
        COUNT(o.order_id) AS orders_count,
        MIN(o.placed_at) AS first_order_at,
        MAX(o.amount) AS only_order_amount
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'captured'
    GROUP BY c.customer_id
),
repeat_cohort AS (
    -- Second order amount for customers who DID return. Falls back to the
    -- overall average order value if fewer than 5 repeat customers exist yet
    -- (early in a fixture run, or a very young merchant) so the query never
    -- divides by a near-empty cohort.
    SELECT AVG(second_order.amount) AS avg_second_order
    FROM (
        SELECT o.customer_id, o.amount,
               ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.placed_at) AS rn
        FROM orders o
        WHERE o.status = 'captured'
    ) second_order
    WHERE second_order.rn = 2
)
SELECT
    cs.customer_id,
    COALESCE(
        (SELECT avg_second_order FROM repeat_cohort WHERE avg_second_order IS NOT NULL),
        (SELECT AVG(amount) FROM orders WHERE status = 'captured')
    ) AS rupees_at_risk
FROM customer_stats cs
WHERE cs.orders_count = 1
  AND CAST(julianday(:now) - julianday(cs.first_order_at) AS INTEGER) >= :first_order_threshold_days;

-- name: cross_sell_affinity
-- Product-pair affinity computed directly from the basket history: how often
-- SKU B appears in the same order as SKU A, as a fraction of all orders
-- containing A. A customer who owns A but has never bought a B with strong
-- affinity to A is the opportunity; rupees at risk is B's list price.
-- Only the single strongest recommendation per customer survives (the window
-- function), so one customer cannot flood the candidate pool with every
-- product they don't yet own.
WITH customer_skus AS (
    SELECT DISTINCT o.customer_id, oi.sku
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.status = 'captured'
),
sku_order_counts AS (
    SELECT oi.sku, COUNT(DISTINCT oi.order_id) AS order_count
    FROM order_items oi
    JOIN orders o ON o.order_id = oi.order_id
    WHERE o.status = 'captured'
    GROUP BY oi.sku
),
pair_counts AS (
    SELECT oi1.sku AS sku_from, oi2.sku AS sku_to, COUNT(DISTINCT oi1.order_id) AS pair_count
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.sku != oi2.sku
    JOIN orders o ON o.order_id = oi1.order_id
    WHERE o.status = 'captured'
    GROUP BY oi1.sku, oi2.sku
),
strong_affinity AS (
    SELECT
        pc.sku_from, pc.sku_to,
        CAST(pc.pair_count AS REAL) / soc.order_count AS confidence
    FROM pair_counts pc
    JOIN sku_order_counts soc ON soc.sku = pc.sku_from
    WHERE pc.pair_count >= :min_pair_count
      AND CAST(pc.pair_count AS REAL) / soc.order_count >= :min_confidence
),
ranked_recommendations AS (
    SELECT
        cs.customer_id,
        sa.sku_to AS recommended_sku,
        p.price AS rupees_at_risk,
        ROW_NUMBER() OVER (PARTITION BY cs.customer_id ORDER BY sa.confidence DESC) AS rn
    FROM customer_skus cs
    JOIN strong_affinity sa ON sa.sku_from = cs.sku
    JOIN products p ON p.sku = sa.sku_to
    WHERE NOT EXISTS (
        SELECT 1 FROM customer_skus cs2
        WHERE cs2.customer_id = cs.customer_id AND cs2.sku = sa.sku_to
    )
)
SELECT customer_id, rupees_at_risk
FROM ranked_recommendations
WHERE rn = 1;
