--EDA
SELECT TOP(10) *
FROM maintenance;

-- Check for Duplicates 
SELECT product_id, COUNT(*) AS cnt
FROM maintenance
GROUP BY product_id
HAVING COUNT(*) > 1;

--  chech for (NULL values):
select count(product_id)  
from maintenance
where product_id is null 

-- # of unique products
SELECT COUNT(DISTINCT product_id) AS unique_products
FROM maintenance;

-- # of failure 
SELECT COUNT(*) AS total_failures
FROM maintenance
WHERE Target = 1 ;

-- total failure for every failure type
SELECT COUNT(*) AS total_failures , failure_type
FROM maintenance
WHERE Target = 1
GROUP BY failure_type ;

-- normal vs failure
SELECT 
    failure_type,
    COUNT(*) AS total_failures,
    min(process_temperature_k) AS min_process_temp,
    min(air_temperature_k) AS min_air_temp,
    min(rotational_speed_rpm) AS min_speed,
    min(torque_nm) AS min_torque
FROM maintenance
GROUP BY failure_type
ORDER BY total_failures DESC;


-- failure ratio
SELECT
round(COUNT(CASE WHEN Target = 1 THEN 1 END) * 100.0 / COUNT(*) ,2)AS failure_rate
FROM maintenance;

--Does the product type affect failure?
SELECT Type, COUNT(*) AS failures
FROM maintenance
WHERE Target = 1
GROUP BY Type;

--TYPE OF FAILURE PER PRODUCT TYPE
SELECT 
    type,
    failure_type,
    COUNT(*) AS total_failures
FROM maintenance
GROUP BY type, failure_type
ORDER BY type, total_failures DESC;

--Ratio of Failure for every product
SELECT 
    type,
    COUNT(CASE WHEN failure_type <> 'No Failure' THEN 1 END) * 100.0 / COUNT(*) AS failure_rate_percent
FROM maintenance
GROUP BY type;

--Avg Tool Wear
select avg(tool_wear_MIN) as avg_ToolWear,[TYPE]
from maintenance
GROUP BY [TYPE]