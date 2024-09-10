use loan_default;

Select * from loan_data

SELECT DISTINCT loan_amount 
FROM loan_data
ORDER BY loan_amount DESC;


ORDER BY loan_amount DESC;
#checking on datatypes
describe Loan_data;

#Getting the first 1000 values
SELECT loan_amount, term, property_value, income, Credit_Score, LTV
FROM loan_data
LIMIT 1000;

#Average loan_amount by gender
SELECT gender, AVG(loan_amount) AS average_loan_amount
FROM loan_data
GROUP BY gender;

#Number of loans grouped by region
SELECT region, COUNT(*) AS loan_count
FROM loan_data
GROUP BY region;

#Top 100 loans with highest amount
SELECT *
FROM loan_data
ORDER BY loan_amount DESC
LIMIT 100;

#Group loan by status and creditWorthines
SELECT Status, Credit_Worthiness, COUNT(*) AS loan_count
FROM loan_data
GROUP BY Status, credit_Worthiness;

#Average interest rate spread by loan type
SELECT loan_type, AVG(interest_rate_spread) AS avg_interest_rate_spread
FROM loan_data
GROUP BY loan_type;

#Loan category based on the loan amount
SELECT id, loan_amount,
  CASE 
    WHEN loan_amount >= 3000000 THEN 'High'
    WHEN loan_amount >= 500000 AND loan_amount < 3000000 THEN 'Medium'
    WHEN loan_amount < 500000 THEN 'Low'
  END AS loan_category
FROM loan_data;

#Loan with high LTV ratios and Low credit score
SELECT id, loan_amount, ltv, credit_score
FROM loan_data
WHERE ltv > 80
  AND credit_score < 600;

#Counting loans with missing income information
SELECT COUNT(*) AS missing_income_count
FROM loan_data
WHERE income IS NULL;

# Grouping and Analyzing Loans based on the loan types and occupancy type
SELECT loan_type, occupancy_type, 
       COUNT(*) AS loan_count, 
       AVG(loan_amount) AS avg_loan_amount
FROM loan_data
GROUP BY loan_type, occupancy_type;

# Loans with upfront charges above average
SELECT *
FROM loan_data
WHERE upfront_charges > (
    SELECT AVG(upfront_charges)
    FROM loan_data
);
