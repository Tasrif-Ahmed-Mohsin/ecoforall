"""
Master Quad-Domain System Audit & Deep Pattern Integration Runner
------------------------------------------------------------------
Executes full end-to-end quad-domain pipeline across Economy, Politics, Environment, and Human/Society:
1. Quad-Domain Dataset Harmonization & Multi-Domain Feature Engineering
2. Bivariate Correlation, Directional Granger Causality & Crisis/Shock Co-occurrence Analysis
3. Walk-Forward Cross-Validation Synergy Tournament & Statistical Significance (Diebold-Mariano) Testing
4. 4D Country-Year Twin (Analog) Matching & Trajectory Pattern Analysis
5. Comprehensive Synthesis of Verified Positive Quad-Domain Findings
"""

import os
import time
import logging
import pandas as pd

from cross_domain_quad_harmonizer import QuadDomainDatasetHarmonizer
import cross_domain_quad_correlation_analyzer as corr_analyzer
import cross_domain_quad_forecasting_evaluator as tourn_evaluator
import cross_domain_quad_pattern_analyzer as pattern_analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def print_banner(title):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80 + "\n")


def run_audit():
    start_time = time.time()

    print_banner("1. Quad-Domain Dataset Harmonization & Feature Engineering")
    harmonizer = QuadDomainDatasetHarmonizer()
    df_panel = harmonizer.run()

    print_banner("2. Bivariate Correlation, Granger Causality & Shocks Analysis")
    corr_df = corr_analyzer.run_bivariate_correlations(df_panel)
    gc_df = corr_analyzer.run_granger_causality(df_panel, max_lag=3)
    co_df = corr_analyzer.run_disaster_crisis_cooccurrence(df_panel)

    print_banner("3. Quad-Domain Walk-Forward Forecasting Synergy Tournament")
    tourn_evaluator.main()

    print_banner("4. 4D Country-Year Twin Pattern Analysis & Positive Findings Synthesis")
    twins_df, _ = pattern_analyzer.run_country_year_twin_pattern_analysis()
    positive_findings_df = pattern_analyzer.synthesize_positive_findings()

    print_banner("5. Executive Master Audit Report: Verified Quad-Domain Empirical Findings")

    print("\n--- ALL VERIFIED POSITIVE EMPIRICAL FINDINGS ---")
    for _, r in positive_findings_df.iterrows():
        print(f"\n[{r['finding_id']}] [{r['category']}] ({r['stat_significance']})")
        print(f" Title: {r['finding_title']}")
        print(f" Empirical Evidence: {r['empirical_evidence']}")

    if twins_df is not None and not twins_df.empty:
        print("\n\n--- 4D COUNTRY-YEAR TWIN MATCHING SAMPLE (Top Historical Analogs) ---")
        cols_to_show = [c for c in ["target_country", "target_year", "twin_rank", "twin_country", "twin_year", "similarity_score", "twin_gdp_growth", "twin_psychology_trust"] if c in twins_df.columns]
        print(twins_df.head(12)[cols_to_show].to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nMaster Quad-Domain Deep Audit completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_audit()
