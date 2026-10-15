-- 08_campaigns.sql
-- Campaign / coupon layer. Campaign membership (campaigns.csv) lists the households TARGETED by each campaign;
-- it is observational: nothing in the data says how households were chosen, so no causal reading is possible.
-- Grain campaign_exposure: one row per (household_id, campaign_id) for households with purchases.
CREATE OR REPLACE TABLE campaign_exposure AS
SELECT c.household_id, c.campaign_id, d.campaign_type, d.start_date, d.end_date
FROM stg_campaigns c
JOIN stg_campaign_descriptions d USING (campaign_id)
WHERE c.household_id IN (SELECT household_id FROM dim_households);

-- Grain coupon_redemptions_clean: one row per redemption by a household with purchases.
CREATE OR REPLACE TABLE coupon_redemptions_clean AS
SELECT r.household_id, r.campaign_id, r.coupon_upc, r.redemption_date
FROM stg_coupon_redemptions r
WHERE r.household_id IN (SELECT household_id FROM dim_households);

-- Grain dim_campaigns: one row per campaign (27).
CREATE OR REPLACE TABLE dim_campaigns AS
WITH tgt AS (
    SELECT campaign_id, COUNT(*) AS n_targeted FROM campaign_exposure GROUP BY 1
),
red AS (
    SELECT campaign_id, COUNT(*) AS n_redemptions, COUNT(DISTINCT household_id) AS n_redeeming_households
    FROM coupon_redemptions_clean GROUP BY 1
),
red_targeted AS (   -- redeemers who were also on the campaign's target list
    SELECT r.campaign_id, COUNT(DISTINCT r.household_id) AS n_redeemers_in_target
    FROM coupon_redemptions_clean r JOIN campaign_exposure e USING (household_id, campaign_id) GROUP BY 1
)
SELECT d.campaign_id, d.campaign_type, d.start_date, d.end_date,
       d.end_date - d.start_date + 1                         AS duration_days,
       COALESCE(t.n_targeted, 0)                             AS n_households_targeted,
       COALESCE(r.n_redeeming_households, 0)                 AS n_redeeming_households,
       COALESCE(r.n_redemptions, 0)                          AS n_redemptions,
       COALESCE(rt.n_redeemers_in_target, 0)                 AS n_redeemers_in_target,
       COALESCE(r.n_redeeming_households, 0) * 1.0 / NULLIF(t.n_targeted, 0) AS redemption_rate_households
FROM stg_campaign_descriptions d
LEFT JOIN tgt t USING (campaign_id)
LEFT JOIN red r USING (campaign_id)
LEFT JOIN red_targeted rt USING (campaign_id)
ORDER BY CAST(d.campaign_id AS INTEGER);
