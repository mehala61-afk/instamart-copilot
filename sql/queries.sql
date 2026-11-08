SELECT city, COUNT(*) AS customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC;

SELECT status, COUNT(*) AS total_orders
FROM orders
GROUP BY status;

SELECT
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_amount
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY total_amount DESC;

SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, total_revenue DESC;

SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;

SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

WITH customer_spending AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT
    customer_id,
    ROUND(total_spend, 2) AS total_spend
FROM customer_spending
WHERE total_spend > (
    SELECT AVG(total_spend)
    FROM customer_spending
)
ORDER BY total_spend DESC;

WITH customer_spending AS (
    SELECT
        c.city,
        o.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS rank_no,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS dense_rank_no,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS row_no
FROM customer_spending
ORDER BY city, rank_no;

WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
        ), 2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
