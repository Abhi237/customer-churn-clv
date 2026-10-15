-- 07_rfm.sql
-- RFM snapshot as of params.rfm_as_of (day after the last transaction). Quintile scores via NTILE.
-- Frequency = shopping days; monetary = merchandise sales dollars (USD).
CREATE OR REPLACE TABLE household_rfm AS
WITH base AS (
    SELECT household_id,
           (SELECT rfm_as_of FROM params) - MAX(visit_date)  AS recency_days,
           COUNT(*)                                          AS frequency,
           SUM(spend)                                        AS monetary
    FROM fct_days
    WHERE visit_date < (SELECT rfm_as_of FROM params)
    GROUP BY household_id
),
scored AS (
    SELECT *,
           6 - NTILE(5) OVER (ORDER BY recency_days, household_id)  AS r_score,
           NTILE(5) OVER (ORDER BY frequency, monetary, household_id) AS f_score,
           NTILE(5) OVER (ORDER BY monetary, household_id)            AS m_score
    FROM base
)
SELECT *,
       CAST(r_score AS VARCHAR) || CAST(f_score AS VARCHAR) || CAST(m_score AS VARCHAR) AS rfm_cell,
       CASE
           WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
           WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
           WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Promising'
           WHEN r_score <= 2 AND f_score >= 3 THEN 'At risk'
           WHEN r_score <= 2 AND f_score <= 2 THEN 'Hibernating'
           ELSE 'Needs attention'
       END AS rfm_segment
FROM scored;
