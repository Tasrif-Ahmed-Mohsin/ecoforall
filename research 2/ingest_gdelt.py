import os
import sys
import yaml
import io
import zipfile
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GDELT_V2_EXPORT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL"
]

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_single_day_gdelt(date_str):
    """Download and process a single daily GDELT ZIP file."""
    url = f"http://data.gdeltproject.org/gdeltv2/{date_str}000000.export.CSV.zip"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df_day = pd.read_csv(f, sep="\t", header=None, names=GDELT_V2_EXPORT_COLUMNS, low_memory=False)
                    
                    df_day = df_day[df_day["Actor1CountryCode"].notnull() & (df_day["Actor1CountryCode"].astype(str).str.len() == 3)].copy()
                    
                    if not df_day.empty:
                        df_day["Day"] = pd.to_datetime(df_day["Day"].astype(str), format="%Y%m%d", errors="coerce")
                        df_day["GoldsteinScale"] = pd.to_numeric(df_day["GoldsteinScale"], errors="coerce")
                        df_day["AvgTone"] = pd.to_numeric(df_day["AvgTone"], errors="coerce")
                        df_day["QuadClass"] = pd.to_numeric(df_day["QuadClass"], errors="coerce")
                        
                        grouped = df_day.groupby(["Day", "Actor1CountryCode"]).agg(
                            goldstein_score_mean=("GoldsteinScale", "mean"),
                            avg_tone_mean=("AvgTone", "mean"),
                            event_count_total=("GlobalEventID", "count"),
                            verbal_coop_count=("QuadClass", lambda x: (x == 1).sum()),
                            material_coop_count=("QuadClass", lambda x: (x == 2).sum()),
                            verbal_conflict_count=("QuadClass", lambda x: (x == 3).sum()),
                            material_conflict_count=("QuadClass", lambda x: (x == 4).sum()),
                            protest_count=("EventRootCode", lambda x: (x.astype(str) == "14").sum())
                        ).reset_index()
                        
                        grouped.rename(columns={"Day": "event_date", "Actor1CountryCode": "country_iso3"}, inplace=True)
                        return len(df_day), grouped
    except Exception:
        pass
    return 0, None

def download_massive_gdelt_data(start_year=2023, num_days=180, max_workers=10):
    """
    Downloads MASSIVE REAL GDELT 2.0 daily event data using parallel multi-threaded HTTP streams.
    """
    logging.info(f"Initiating MASSIVE parallel download of {num_days} days of real GDELT archives ({max_workers} threads)...")
    
    start_date = datetime(start_year, 1, 1)
    dates = [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range(num_days)]
    
    daily_records = []
    total_raw_events = 0
    completed_days = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(download_single_day_gdelt, d): d for d in dates}
        for future in as_completed(future_to_date):
            raw_count, grouped_df = future.result()
            completed_days += 1
            if grouped_df is not None:
                total_raw_events += raw_count
                daily_records.append(grouped_df)
                if completed_days % 20 == 0 or completed_days == num_days:
                    logging.info(f"Progress: {completed_days}/{num_days} days processed | Total raw events downloaded: {total_raw_events:,}")
                    
    if daily_records:
        combined = pd.concat(daily_records, ignore_index=True)
        logging.info(f"MASSIVE DOWNLOAD COMPLETE! Downloaded {total_raw_events:,} raw GDELT events across {len(combined):,} daily country records!")
        return combined
    return None

def process_and_save_massive_panel(raw_df, frequency="1W"):
    if raw_df is None or raw_df.empty:
        logging.error("Cannot process empty DataFrame.")
        return
        
    logging.info(f"Resampling MASSIVE GDELT panel to frequency: {frequency}...")
    raw_df["event_date"] = pd.to_datetime(raw_df["event_date"])
    
    aggregated = []
    for iso3, group in raw_df.groupby("country_iso3"):
        group = group.set_index("event_date").sort_index()
        
        resampled = group.resample(frequency).agg({
            "goldstein_score_mean": "mean",
            "avg_tone_mean": "mean",
            "event_count_total": "sum",
            "verbal_coop_count": "sum",
            "material_coop_count": "sum",
            "verbal_conflict_count": "sum",
            "material_conflict_count": "sum",
            "protest_count": "sum",
        }).reset_index()
        
        resampled["country_iso3"] = iso3
        
        total_coop = resampled["verbal_coop_count"] + resampled["material_coop_count"]
        total_conf = resampled["verbal_conflict_count"] + resampled["material_conflict_count"]
        resampled["conflict_cooperation_ratio"] = (total_conf + 1.0) / (total_coop + 1.0)
        resampled["conflict_intensity_pct"] = total_conf / (resampled["event_count_total"] + 1e-5)
        
        aggregated.append(resampled)
        
    final_df = pd.concat(aggregated, ignore_index=True)
    final_df.rename(columns={"event_date": "timestamp"}, inplace=True)
    
    os.makedirs("data", exist_ok=True)
    
    csv_path = "data/massive_real_gdelt_panel.csv"
    parquet_path = "data/massive_real_gdelt_panel.parquet"
    
    final_df.to_csv(csv_path, index=False)
    final_df.to_parquet(parquet_path, index=False)
    final_df.to_csv("data/gdelt_panel_wide.csv", index=False)
    
    logging.info(f"MASSIVE GDELT Panel saved to {csv_path} ({len(final_df):,} rows across {final_df['country_iso3'].nunique()} countries)")
    return final_df

def main():
    config = load_config()
    freq = config["domain"]["frequency"]
    
    # Download 180 days (6 months) of REAL GDELT 2.0 events in parallel (10 threads)
    raw_df = download_massive_gdelt_data(start_year=2023, num_days=180, max_workers=10)
    
    if raw_df is not None:
        process_and_save_massive_panel(raw_df, frequency=freq)
        print("\n" + "="*70)
        print(" SUCCESS: MASSIVE REAL GDELT DATASET DOWNLOADED AND EXTRACTED!")
        print("="*70)

if __name__ == "__main__":
    main()
