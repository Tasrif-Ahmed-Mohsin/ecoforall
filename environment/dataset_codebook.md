# 65-Year Environmental & Climate Risk Dataset Codebook (1960–2025)

## Overview

This dataset codebook documents the canonical multi-variate panel schema for the **Universal Dynamic-Horizon Environment & Climate Risk Engine**.

* **Temporal Coverage**: 1960 to 2025 (65 Years)
* **Entities**: ~200 ISO3 Countries & Global Regions
* **Frequency**: Annual (`1Y`) & Monthly (`1M`)
* **Canonical Schema**: `[iso3, year, indicator_id, value]` -> Pivoted Panel `[iso3, year, var1, var2, ...]`

---

## Key Indicators & Metadata

| Indicator Code | Full Indicator Name | Unit | Primary Source | Expected Range / Property |
| :--- | :--- | :--- | :--- | :--- |
| `co2_emissions_per_capita` | Carbon Dioxide Emissions per Capita | Metric Tons per Person | World Bank WDI / Global Carbon Project | $[0.0, 60.0]$ |
| `temp_anomaly_celsius` | Surface Temperature Anomaly relative to 1951-1980 Baseline | °C | NOAA GISTEMP v4 / ERA5 | $[-2.5, +4.5]$ |
| `forest_area_pct_land` | Forest Area as Percent of Total Land Area | % | FAOSTAT / World Bank WDI | $[0.0, 100.0]$ |
| `extreme_disasters_count` | Frequency of Extreme Climate Disasters (Floods, Droughts, Wildfires, Storms) | Count per Year | EM-DAT (CRED) | $[0, 150]$ |
| `renewable_energy_pct_share` | Renewable Energy Consumption as Percent of Total Final Energy Consumption | % | World Bank WDI / IEA | $[0.0, 100.0]$ |
| `greenhouse_gas_total_kt` | Total Greenhouse Gas Emissions | Thousand Metric Tons $CO_2$ eq | World Bank WDI / CAIT | $[10^2, 1.5 \times 10^7]$ |
| `energy_use_per_capita` | Energy Use per Person | kg oil equivalent | World Bank WDI | $[100, 20000]$ |
| `freshwater_withdrawal_pct` | Annual Freshwater Withdrawals as % of Internal Resources | % | FAOSTAT / World Bank WDI | $[0.0, 1000.0]$ |
| `agricultural_land_pct` | Agricultural Land Area | % of Land Area | FAOSTAT / World Bank WDI | $[0.0, 95.0]$ |
| `protected_area_pct` | Terrestrial & Marine Protected Areas | % of Total Area | UNEP-WCMC / World Bank | $[0.0, 60.0]$ |

---

## Data Harmonization Rules

1. **Entity Identifier**: Standardized 3-letter ISO 3166-1 alpha-3 code (`iso3`).
2. **Missing Data Imputation**: Linear interpolation inside historical series per entity; forward filling for short gaps; median indicator imputation across regional peers for remaining missing values.
3. **Rank Percentile Transformation**: Continuous indicators are scaled to $[0, 1]$ uniform rank percentiles across entities per year slice before state vector similarity retrieval to eliminate units and ensure scale-invariant matching.
