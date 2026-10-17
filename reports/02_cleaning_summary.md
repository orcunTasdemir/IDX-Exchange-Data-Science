# Week 3 Cleaning: Short Summary

**IDX Exchange Data Science Internship** · Orcun Tasdemir · Full write-up: [`02_preprocessing_memo.md`](02_preprocessing_memo.md)

**In one sentence:** the data was cut to single-family homes, 873 impossible or duplicate sales (0.28%) were removed, 431 impossible single values were blanked and filled in later, every column that reveals the sale price was dropped, and extreme prices were trimmed using cutoffs from the training months only.

## 1. Single-family homes only

| Step | Rows left |
|---|---:|
| All closed sales (28 monthly files, Jan 2024 to Apr 2026) | 615,707 |
| Residential only | 414,054 |
| Single-family homes only | **309,735** |

## 2. Removed sales (873 rows, 0.28%)

| Rule | Removed |
|---|---:|
| Exact duplicate rows | 29 |
| Older copies of the same listing (latest kept) | 235 |
| Missing or $0 price | 2 |
| Recorded outside California | 22 |
| Closed before the listing started | 44 |
| Closed before the purchase contract was signed | 194 |
| Living area missing, under 300 or over 20,000 sq ft | 328 |
| More than 15 bedrooms or bathrooms | 19 |
| **Left after cleaning** | **308,862** |

Repeated listing IDs turned out to be the same house recorded again in a later monthly file, so the latest record was kept. This was done before splitting by month, so that one sale cannot appear in both the training and test data.

## 3. Bad single values blanked, not whole rows

431 impossible values were set to missing and filled in later: lots of 0 sq ft (162), 0 bedrooms (121), 0 bathrooms (78), coordinates outside California (62) and impossible build years (8). Gaps were also filled from columns recording the same fact: 5,819 story counts from the `Levels` field and 20 lot sizes from lot size in acres.

## 4. Columns: 17 inputs

- **Kept:** living area, lot size, bedrooms, bathrooms, year built, garage spaces, stories, monthly association fee, latitude, longitude, five yes/no facts (pool, view, attached garage, fireplace, newly built), county and ZIP code.
- **Removed because they reveal the sale price:** list price, original list price, days on market and buyer-agent compensation. They are set during the sale, so they do not exist for homes that are not for sale.
- **Also removed:** empty or mostly empty columns, agent and office names, ID fields, and columns repeating a kept fact.

## 5. Extreme prices trimmed, using training months only

Sales outside the 0.5th to 99.5th percentile of price and of price per square foot were removed. The cutoffs come from the 12 training months only and are applied unchanged to the test month.

| | Lower | Upper |
|---|---:|---:|
| Price | $185,000 | $8,600,000 |
| Price per sq ft | $144 | $2,299 |

This removed 1.50% of training sales and 1.66% of test sales.

## 6. Remaining gaps, filled from training data only

- **Numbers:** training median, plus a 0/1 column marking each filled value.
- **Coordinates:** median location of the home's ZIP code.
- **Yes/no facts:** missing counts as "no", plus a marker column.
- **County:** one 0/1 column per county. **ZIP code:** the average log sale price of training homes in that ZIP code, computed without the home's own price.

**Why training data only:** if any medians, cutoffs or averages used the test month, the test score would be inflated and would not show how the model does on genuinely new sales.
