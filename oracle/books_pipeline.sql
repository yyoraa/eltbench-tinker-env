-- Reference solution for snowflake/books. Never copied into a task image or tests/.
-- Reads the raw tables in the session schema and rebuilds all three targets.
CREATE OR REPLACE TABLE PUBLISHERS AS
WITH avg_pages AS (SELECT AVG(num_pages) AS a FROM BOOK),
ranked AS (
  SELECT b.*,
    ROW_NUMBER() OVER (PARTITION BY publisher_id ORDER BY publication_date ASC, title ASC) AS oldest_rank,
    ROW_NUMBER() OVER (PARTITION BY publisher_id ORDER BY num_pages DESC, title DESC) AS longest_rank
  FROM BOOK b
)
SELECT p.publisher_id,
  p.publisher_name,
  COUNT(b.book_id) AS num_books,
  COALESCE(SUM(CASE WHEN b.num_pages > 0.7 * (SELECT a FROM avg_pages) THEN 1 ELSE 0 END), 0)
    AS num_books_pages_greater_than_70_per_of_avg,
  100.0 * SUM(CASE WHEN l.language_name = 'Japanese' THEN 1 ELSE 0 END) / NULLIF(COUNT(b.book_id), 0)
    AS per_japanese_books,
  MAX(CASE WHEN b.oldest_rank = 1 THEN b.title END) AS title_of_the_oldest_book,
  MAX(CASE WHEN b.longest_rank = 1 THEN b.title END) AS book_title_with_the_most_pages
FROM PUBLISHER p
LEFT JOIN ranked b ON b.publisher_id = p.publisher_id
LEFT JOIN BOOK_LANGUAGE l ON l.language_id = b.language_id
GROUP BY p.publisher_id, p.publisher_name;

CREATE OR REPLACE TABLE AUTHORS AS
WITH authored AS (
  SELECT ba.author_id, b.*
  FROM BOOK_AUTHOR ba JOIN BOOK b ON b.book_id = ba.book_id
),
first_books AS (
  SELECT author_id, title FROM authored
  QUALIFY ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY publication_date ASC, title ASC) = 1
),
pages AS (SELECT author_id, SUM(num_pages) AS total_pages FROM authored GROUP BY author_id),
sales AS (
  SELECT a.author_id, a.title, ol.price
  FROM authored a JOIN ORDER_LINE ol ON ol.book_id = a.book_id
),
priciest AS (
  SELECT author_id, title FROM sales
  QUALIFY ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY price DESC, title DESC) = 1
),
avg_price AS (SELECT author_id, AVG(price) AS price FROM sales GROUP BY author_id),
published_1992 AS (
  SELECT author_id,
    100.0 * SUM(CASE WHEN LEFT(publication_date, 4) = '1992' THEN 1 ELSE 0 END) / COUNT(*) AS pct
  FROM authored GROUP BY author_id
)
SELECT a.author_id,
  a.author_name,
  f.title AS first_book,
  y.pct AS per_book_published_1992,
  CASE WHEN t.total_pages < (SELECT AVG(total_pages) FROM pages) THEN 1 ELSE 0 END
    AS wrote_fewer_total_pages_than_the_average,
  m.title AS most_expensive_book,
  CASE WHEN v.price > 19 THEN 1 ELSE 0 END AS avg_book_price_over_19
FROM AUTHOR a
LEFT JOIN first_books f ON f.author_id = a.author_id
LEFT JOIN pages t ON t.author_id = a.author_id
LEFT JOIN priciest m ON m.author_id = a.author_id
LEFT JOIN avg_price v ON v.author_id = a.author_id
LEFT JOIN published_1992 y ON y.author_id = a.author_id;

CREATE OR REPLACE TABLE CUSTOMERS AS
WITH shipping AS (
  SELECT o.customer_id, s.method_name
  FROM CUST_ORDER o
  JOIN SHIPPING_METHOD s ON TRY_TO_NUMBER(s.method_id) = o.shipping_method_id
),
lines AS (
  SELECT o.customer_id, ol.price, ol.book_id
  FROM CUST_ORDER o JOIN ORDER_LINE ol ON ol.order_id = o.order_id
),
expensive AS (
  SELECT customer_id, 100.0 * SUM(CASE WHEN price > 13 THEN 1 ELSE 0 END) / COUNT(*) AS pct
  FROM lines GROUP BY customer_id
),
oldest AS (SELECT book_id FROM BOOK ORDER BY publication_date ASC, title ASC LIMIT 1),
preferred AS (
  SELECT customer_id, method_name FROM (
    SELECT customer_id, method_name, COUNT(*) AS uses FROM shipping GROUP BY customer_id, method_name
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY uses DESC, method_name ASC) = 1
),
international AS (
  SELECT customer_id,
    100.0 * SUM(CASE WHEN method_name = 'International' THEN 1 ELSE 0 END) / COUNT(*) AS pct
  FROM shipping GROUP BY customer_id
),
order_counts AS (SELECT customer_id, COUNT(*) AS orders FROM CUST_ORDER GROUP BY customer_id),
bought_oldest AS (
  SELECT DISTINCT customer_id FROM lines WHERE book_id IN (SELECT book_id FROM oldest)
)
SELECT c.customer_id,
  c.first_name || ' ' || c.last_name AS customer_name,
  c.email,
  e.pct AS per_books_over_13,
  COALESCE(n.orders, 0) AS num_orders,
  p.method_name AS preferred_shipping_method,
  CASE WHEN b.customer_id IS NULL THEN 0 ELSE 1 END AS has_ordered_the_oldest_book,
  i.pct AS per_order_shipped_internationally
FROM CUSTOMER c
LEFT JOIN expensive e ON e.customer_id = c.customer_id
LEFT JOIN order_counts n ON n.customer_id = c.customer_id
LEFT JOIN preferred p ON p.customer_id = c.customer_id
LEFT JOIN bought_oldest b ON b.customer_id = c.customer_id
LEFT JOIN international i ON i.customer_id = c.customer_id;
