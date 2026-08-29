# MACHINE-GENERATED REAL CROSS-DOMAIN DATA DICTIONARY
**Audit Date:** 2026-08-29
**Panel Dimensions:** 15071 rows × 246 columns
**Coverage:** 237 countries, 1960 to 2024

## Domain Allocations & Provenance

| Domain | Features | Primary Public Sources | Provenance Type |
|---|---|---|---|
| **1. Macro/Trade** | 209 | Global Macro Database (GMD v6), World Bank, IMF WEO, OECD | **REAL (Verified)** |
| **2. Politics/Institutions** | 25 | Varieties of Democracy (V-Dem v14), Our World In Data | **REAL (Verified)** |
| **3. Climate/Environment** | 10 | Copernicus ERA5 / Berkeley Earth, Global Carbon Budget | **REAL (Verified)** |
| **4. Targets/Meta** | 3 | Constructed Forward GDP per Capita Growth ($h=1,3,5$) | **DERIVED TARGETS** |

## Complete Variable Manifest

| Variable Name | Non-Null Count | Data Type | Description |
|---|---|---|---|
| `iso3` | 15,071 | `object` | Panel index identifier |
| `year` | 15,071 | `int64` | Panel index identifier |
| `banking_crisis` | 8,553 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate` | 6,379 | `float64` | Macroeconomic GMD indicator |
| `cgov_debt_gdp` | 8,106 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp` | 12,534 | `float64` | Macroeconomic GMD indicator |
| `cpi` | 11,605 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis` | 8,396 | `float64` | Macroeconomic GMD indicator |
| `current_account` | 9,099 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp` | 9,099 | `float64` | Macroeconomic GMD indicator |
| `current_account_usd` | 9,082 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp` | 12,880 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp` | 12,207 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd` | 13,904 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator` | 13,020 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal` | 13,261 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_usd` | 13,015 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real` | 13,306 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd` | 12,670 | `float64` | Macroeconomic GMD indicator |
| `gdp_real` | 13,330 | `float64` | Macroeconomic GMD indicator |
| `gdp_real_usd` | 12,670 | `float64` | Macroeconomic GMD indicator |
| `gen_gov_debt_gdp` | 9,259 | `float64` | Macroeconomic GMD indicator |
| `gen_gov_deficit_gdp` | 7,709 | `float64` | Macroeconomic GMD indicator |
| `gen_gov_tax_gdp` | 3,521 | `float64` | Macroeconomic GMD indicator |
| `gov_consumption_gdp` | 12,370 | `float64` | Macroeconomic GMD indicator |
| `gov_debt` | 9,437 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp` | 9,437 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit` | 9,399 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp` | 9,399 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure` | 9,676 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp` | 9,676 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue` | 9,604 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp` | 9,604 | `float64` | Macroeconomic GMD indicator |
| `gov_tax` | 5,635 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp` | 5,635 | `float64` | Macroeconomic GMD indicator |
| `house_price_index` | 2,352 | `float64` | Macroeconomic GMD indicator |
| `household_consumption_gdp` | 12,464 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp` | 12,872 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate` | 11,453 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp` | 12,561 | `float64` | Macroeconomic GMD indicator |
| `long_rate` | 2,771 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m0` | 7,811 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1` | 8,017 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2` | 7,472 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3` | 4,832 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m4` | 142 | `float64` | Macroeconomic GMD indicator |
| `population` | 14,678 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index` | 2,342 | `float64` | Macroeconomic GMD indicator |
| `reer` | 10,033 | `float64` | Macroeconomic GMD indicator |
| `short_rate` | 5,143 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis` | 8,353 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate` | 6,290 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc` | 13,306 | `float64` | Macroeconomic GMD indicator |
| `banking_crisis_lag1` | 8,553 | `float64` | Macroeconomic GMD indicator |
| `banking_crisis_lag5` | 8,535 | `float64` | Macroeconomic GMD indicator |
| `banking_crisis_delta5` | 7,735 | `float64` | Macroeconomic GMD indicator |
| `banking_crisis_logret5` | 364 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate_lag1` | 6,276 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate_lag5` | 5,799 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate_delta5` | 5,464 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate_logret5` | 5,297 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp_lag1` | 12,321 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp_lag5` | 11,462 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp_delta5` | 11,434 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp_logret5` | 11,434 | `float64` | Macroeconomic GMD indicator |
| `cpi_lag1` | 11,407 | `float64` | Macroeconomic GMD indicator |
| `cpi_lag5` | 10,576 | `float64` | Macroeconomic GMD indicator |
| `cpi_delta5` | 10,498 | `float64` | Macroeconomic GMD indicator |
| `cpi_logret5` | 10,498 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis_lag1` | 8,396 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis_lag5` | 8,396 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis_delta5` | 7,596 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis_logret5` | 492 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp_lag1` | 8,906 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp_lag5` | 8,114 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp_delta5` | 8,053 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp_logret5` | 6,098 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp_lag1` | 12,667 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp_lag5` | 11,792 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp_delta5` | 11,744 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp_logret5` | 11,744 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp_lag1` | 11,996 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp_lag5` | 11,148 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp_delta5` | 11,122 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp_logret5` | 11,121 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd_lag1` | 13,672 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd_lag5` | 12,744 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd_delta5` | 12,728 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd_logret5` | 12,728 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator_lag1` | 12,807 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator_lag5` | 11,933 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator_delta5` | 11,875 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator_logret5` | 11,875 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_lag1` | 13,047 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_lag5` | 12,168 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_delta5` | 12,111 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_logret5` | 12,111 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_lag1` | 13,093 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_lag5` | 12,219 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_delta5` | 12,166 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_logret5` | 12,166 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd_lag1` | 12,458 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd_lag5` | 11,592 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd_delta5` | 11,580 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd_logret5` | 11,580 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp_lag1` | 9,244 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp_lag5` | 8,474 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp_delta5` | 8,402 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp_logret5` | 8,402 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp_lag1` | 9,204 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp_lag5` | 8,421 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp_delta5` | 8,283 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp_logret5` | 6,207 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp_lag1` | 9,481 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp_lag5` | 8,695 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp_delta5` | 8,573 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp_logret5` | 8,573 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp_lag1` | 9,409 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp_lag5` | 8,626 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp_delta5` | 8,500 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp_logret5` | 8,500 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp_lag1` | 5,556 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp_lag5` | 5,029 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp_delta5` | 4,633 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp_logret5` | 4,633 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp_lag1` | 12,659 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp_lag5` | 11,784 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp_delta5` | 11,736 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp_logret5` | 11,736 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate_lag1` | 11,253 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate_lag5` | 10,417 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate_delta5` | 10,347 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate_logret5` | 9,260 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp_lag1` | 12,348 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp_lag5` | 11,492 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp_delta5` | 11,466 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp_logret5` | 11,458 | `float64` | Macroeconomic GMD indicator |
| `long_rate_lag1` | 2,707 | `float64` | Macroeconomic GMD indicator |
| `long_rate_lag5` | 2,446 | `float64` | Macroeconomic GMD indicator |
| `long_rate_delta5` | 2,340 | `float64` | Macroeconomic GMD indicator |
| `long_rate_logret5` | 2,302 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1_lag1` | 7,977 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1_lag5` | 7,669 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1_delta5` | 7,161 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1_logret5` | 7,161 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2_lag1` | 7,453 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2_lag5` | 7,229 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2_delta5` | 6,663 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2_logret5` | 6,663 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3_lag1` | 4,709 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3_lag5` | 4,133 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3_delta5` | 4,008 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3_logret5` | 4,008 | `float64` | Macroeconomic GMD indicator |
| `population_lag1` | 14,454 | `float64` | Macroeconomic GMD indicator |
| `population_lag5` | 13,538 | `float64` | Macroeconomic GMD indicator |
| `population_delta5` | 13,527 | `float64` | Macroeconomic GMD indicator |
| `population_logret5` | 13,527 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index_lag1` | 2,285 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index_lag5` | 2,053 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index_delta5` | 2,052 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index_logret5` | 2,052 | `float64` | Macroeconomic GMD indicator |
| `reer_lag1` | 9,921 | `float64` | Macroeconomic GMD indicator |
| `reer_lag5` | 9,213 | `float64` | Macroeconomic GMD indicator |
| `reer_delta5` | 9,132 | `float64` | Macroeconomic GMD indicator |
| `reer_logret5` | 9,132 | `float64` | Macroeconomic GMD indicator |
| `short_rate_lag1` | 5,057 | `float64` | Macroeconomic GMD indicator |
| `short_rate_lag5` | 4,640 | `float64` | Macroeconomic GMD indicator |
| `short_rate_delta5` | 4,433 | `float64` | Macroeconomic GMD indicator |
| `short_rate_logret5` | 4,244 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis_lag1` | 8,353 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis_lag5` | 8,351 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis_delta5` | 7,553 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis_logret5` | 265 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate_lag1` | 6,154 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate_lag5` | 5,562 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate_delta5` | 5,091 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate_logret5` | 5,091 | `float64` | Macroeconomic GMD indicator |
| `vdem_electoral_democracy` | 11,181 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_electoral_democracy_lag1` | 11,007 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_electoral_democracy_lag5` | 10,311 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_electoral_democracy_delta1` | 11,006 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_liberal_democracy` | 11,097 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_liberal_democracy_lag1` | 10,923 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_liberal_democracy_lag5` | 10,227 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_liberal_democracy_delta1` | 10,920 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_political_corruption` | 10,458 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_political_corruption_lag1` | 10,284 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_political_corruption_lag5` | 9,588 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_political_corruption_delta1` | 10,284 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_rule_of_law` | 11,166 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_rule_of_law_lag1` | 10,992 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_rule_of_law_lag5` | 10,296 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_rule_of_law_delta1` | 10,992 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_freedom_expression` | 11,182 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_freedom_expression_lag1` | 11,008 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_freedom_expression_lag5` | 10,312 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_freedom_expression_delta1` | 11,008 | `float64` | V-Dem v14 institutional/democracy indicator |
| `climate_temperature_anomaly` | 12,347 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_temperature_anomaly_lag1` | 12,156 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_temperature_anomaly_lag5` | 11,392 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_temperature_anomaly_delta1` | 12,156 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_annual_co2` | 13,216 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_annual_co2_lag1` | 13,003 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_annual_co2_lag5` | 12,158 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_annual_co2_delta1` | 13,002 | `float64` | ERA5 / Carbon Budget climate indicator |
| `banking_crisis_roll5_mean` | 8,881 | `float64` | Macroeconomic GMD indicator |
| `central_bank_rate_roll5_mean` | 6,454 | `float64` | Macroeconomic GMD indicator |
| `consumption_gdp_roll5_mean` | 12,332 | `float64` | Macroeconomic GMD indicator |
| `cpi_roll5_mean` | 11,446 | `float64` | Macroeconomic GMD indicator |
| `currency_crisis_roll5_mean` | 8,716 | `float64` | Macroeconomic GMD indicator |
| `current_account_gdp_roll5_mean` | 8,934 | `float64` | Macroeconomic GMD indicator |
| `exports_gdp_roll5_mean` | 12,687 | `float64` | Macroeconomic GMD indicator |
| `fixed_investment_gdp_roll5_mean` | 12,006 | `float64` | Macroeconomic GMD indicator |
| `fx_to_usd_roll5_mean` | 13,687 | `float64` | Macroeconomic GMD indicator |
| `gdp_deflator_roll5_mean` | 12,831 | `float64` | Macroeconomic GMD indicator |
| `gdp_nominal_roll5_mean` | 13,071 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_roll5_mean` | 13,114 | `float64` | Macroeconomic GMD indicator |
| `gdp_pc_real_usd_roll5_mean` | 12,464 | `float64` | Macroeconomic GMD indicator |
| `gov_debt_gdp_roll5_mean` | 9,292 | `float64` | Macroeconomic GMD indicator |
| `gov_deficit_gdp_roll5_mean` | 9,301 | `float64` | Macroeconomic GMD indicator |
| `gov_expenditure_gdp_roll5_mean` | 9,550 | `float64` | Macroeconomic GMD indicator |
| `gov_revenue_gdp_roll5_mean` | 9,492 | `float64` | Macroeconomic GMD indicator |
| `gov_tax_gdp_roll5_mean` | 5,746 | `float64` | Macroeconomic GMD indicator |
| `imports_gdp_roll5_mean` | 12,679 | `float64` | Macroeconomic GMD indicator |
| `inflation_rate_roll5_mean` | 11,296 | `float64` | Macroeconomic GMD indicator |
| `investment_gdp_roll5_mean` | 12,358 | `float64` | Macroeconomic GMD indicator |
| `long_rate_roll5_mean` | 2,763 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m1_roll5_mean` | 8,201 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m2_roll5_mean` | 7,703 | `float64` | Macroeconomic GMD indicator |
| `money_supply_m3_roll5_mean` | 4,765 | `float64` | Macroeconomic GMD indicator |
| `population_roll5_mean` | 14,459 | `float64` | Macroeconomic GMD indicator |
| `real_house_price_index_roll5_mean` | 2,285 | `float64` | Macroeconomic GMD indicator |
| `reer_roll5_mean` | 9,928 | `float64` | Macroeconomic GMD indicator |
| `short_rate_roll5_mean` | 5,162 | `float64` | Macroeconomic GMD indicator |
| `sov_debt_crisis_roll5_mean` | 8,673 | `float64` | Macroeconomic GMD indicator |
| `unemployment_rate_roll5_mean` | 6,338 | `float64` | Macroeconomic GMD indicator |
| `vdem_electoral_democracy_roll5_mean` | 11,008 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_liberal_democracy_roll5_mean` | 10,927 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_political_corruption_roll5_mean` | 10,284 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_rule_of_law_roll5_mean` | 10,992 | `float64` | V-Dem v14 institutional/democracy indicator |
| `vdem_freedom_expression_roll5_mean` | 11,008 | `float64` | V-Dem v14 institutional/democracy indicator |
| `climate_temperature_anomaly_roll5_mean` | 12,156 | `float64` | ERA5 / Carbon Budget climate indicator |
| `climate_annual_co2_roll5_mean` | 13,006 | `float64` | ERA5 / Carbon Budget climate indicator |
| `gdp_pc_growth_1y_fwd` | 13,077 | `float64` | Forward-looking evaluation target |
| `gdp_pc_growth_3y_fwd` | 12,621 | `float64` | Forward-looking evaluation target |
| `gdp_pc_growth_5y_fwd` | 12,166 | `float64` | Forward-looking evaluation target |