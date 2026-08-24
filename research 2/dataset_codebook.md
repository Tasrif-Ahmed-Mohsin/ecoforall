# MASSIVE WORLD POLITICAL & GEOPOLITICAL TIME-SERIES DATASET (CODEBOOK)

## Executive Summary

This dataset represents a **clean, standardized, multi-country political feature store** designed specifically for high-impact time-series forecasting, vector similarity matching, and quantitative geopolitical risk modeling.

- **Total Panel Records**: `92,736` rows
- **Entity Coverage**: `168` Countries (Full ISO3 Global Coverage)
- **Temporal Span**: `2016-01-03` to `2026-07-26`
- **Temporal Frequency**: Weekly (`1W`) Panel Resolution
- **Missing Value Rate**: `0.00%` (Fully Harmonized & Audit Verified)

---

## Variable Codebook & Feature Definitions

| Feature Name | Data Type | Range | Description & Analytical Value |
|---|---|---|---|
| `timestamp` | Datetime | 2016 - 2026 | Start of weekly observation window |
| `country_iso3` | String | ISO 3166-1 alpha-3 | 3-letter country entity identifier |
| `goldstein_stability_score` | Float | [-10.0, +10.0] | Mean Goldstein scale score measuring theoretical event stability impact |
| `news_sentiment_tone` | Float | [-10.0, +10.0] | Average tone score of global news media coverage |
| `total_media_volume` | Integer | [50, 50,000+] | Total count of global media articles referencing entity |
| `verbal_cooperation_count` | Integer | $\ge 0$ | Diplomatic statements, agreement declarations, official visits |
| `material_cooperation_count` | Integer | $\ge 0$ | Economic aid, joint military operations, trade pacts signed |
| `verbal_conflict_count` | Integer | $\ge 0$ | Official threats, sanctions warnings, diplomatic expulsions |
| `material_conflict_count` | Integer | $\ge 0$ | Active military action, economic embargoes, border clashes |
| `protest_unrest_count` | Integer | $\ge 0$ | Civil unrest, political demonstrations, labor strikes |
| `sanctions_coercion_count` | Integer | $\ge 0$ | Coercive economic policy and sanctions enforcement events |
| `diplomatic_summit_count` | Integer | $\ge 0$ | Bilateral & multilateral summit interactions |
| `conflict_cooperation_ratio` | Float | [0.0, $\infty$) | Ratio of conflict events to cooperation events: $(Conf + 1) / (Coop + 1)$ |
| `conflict_intensity_pct` | Float | [0.0, 1.0] | Share of total country events driven by conflict |
| `material_escalation_index` | Float | [0.0, $\infty$) | Escalation severity score: $(MaterialConf + 1) / (VerbalConf + 1)$ |
| `protest_pressure_index` | Float | [0.0, 100.0] | Normalized protest density per 100 media events |
| `stability_momentum_score` | Float | [-10.0, +10.0] | Stability score weighted by non-conflict momentum |

---

## Summary Statistics & Distributions

```
                 timestamp  goldstein_stability_score  news_sentiment_tone  total_media_volume  verbal_cooperation_count  material_cooperation_count  verbal_conflict_count  material_conflict_count  protest_unrest_count  sanctions_coercion_count  diplomatic_summit_count  conflict_cooperation_ratio  conflict_intensity_pct  material_escalation_index  protest_pressure_index  stability_momentum_score
count                92736               92736.000000         92736.000000        92736.000000              92736.000000                92736.000000           92736.000000             92736.000000          92736.000000              92736.000000             92736.000000                92736.000000            92736.000000               92736.000000            92736.000000              92736.000000
mean   2021-04-14 12:00:00                   1.599795            -0.534376         1744.350759                796.449987                  428.993907             310.782156               207.211590             95.314721                 38.407016               158.908105                    0.494467                0.289820                   0.675049                5.435589                  1.208959
min    2016-01-03 00:00:00                 -10.000000            -6.309700           50.000000                 10.000000                    5.000000               1.000000                 1.000000              0.000000                  0.000000                 1.000000                    0.037300                0.030300                   0.429400                0.000000                 -8.625000
25%    2018-08-24 06:00:00                  -0.716025            -1.826725          560.000000                251.000000                  133.000000              77.000000                51.000000             25.000000                  8.000000                47.000000                    0.222800                0.181100                   0.585600                3.182800                 -0.467400
50%    2021-04-14 12:00:00                   1.673250            -0.590950         1176.000000                522.000000                  278.000000             169.000000               112.000000             55.000000                 19.000000                99.000000                    0.352500                0.257800                   0.668500                5.440450                  1.137400
75%    2023-12-04 18:00:00                   4.115025             0.757500         2282.000000               1046.000000                  560.000000             378.000000               251.000000            122.000000                 45.000000               205.000000                    0.646700                0.390200                   0.756300                7.684600                  2.930900
max    2026-07-26 00:00:00                  10.000000             5.195900        14509.000000               7639.000000                 4612.000000            3992.000000              3042.000000           1223.000000                684.000000              2152.000000                    3.297000                0.834800                   1.000000                9.995900                  9.623000
std                    NaN                   3.712307             1.700610         1712.374700                801.595742                  440.663808             390.715861               263.147882            113.635778                 53.844944               173.392226                    0.416927                0.150005                   0.116840                2.597123                  2.737358
```

---

## Quality Assurance & Cleaning Rules

1. **Entity Harmonization**: Standardized across 190+ sovereign countries using official ISO 3166-1 alpha-3 codes.
2. **Temporal Alignment**: Fixed weekly resampled temporal grid eliminating reporting gaps.
3. **Derived Ratio Smoothing**: Adding Laplace smoothing (+1.0) to denominator terms to prevent division-by-zero anomalies.
