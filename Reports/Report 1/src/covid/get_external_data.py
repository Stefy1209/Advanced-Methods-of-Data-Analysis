"""Download and filter the external data used for the validation step (notebook Step 11).

Run from Reports/Report 1/src/covid/:   python get_external_data.py
Writes ../../data/covid/external/owid_daily_europe_2020.csv and owid_excess_mortality_europe_2020.csv

Sources (Our World in Data, CC BY 4.0), downloaded 19 Nov 2026:
- https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
  (testing: OWID testing dataset; stringency: Oxford COVID-19 Government Response Tracker, Hale et al. 2021)
- https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/excess_mortality/excess_mortality.csv
  (Human Mortality Database STMF, World Mortality Dataset (Karlinsky & Kobak 2021), Eurostat; baseline = projection
  from 2015-2019, Karlinsky & Kobak method)
"""
from pathlib import Path
import io
import urllib.request
import pandas as pd

OUT = Path("../../data/covid/external")
OUT.mkdir(parents=True, exist_ok=True)
URL_DAILY = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
URL_EXCESS = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/excess_mortality/excess_mortality.csv"

# ECDC "Europe" names (data.csv) -> OWID location names
ECDC_EUROPE = ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia_and_Herzegovina",
               "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Faroe_Islands", "Finland", "France",
               "Georgia", "Germany", "Gibraltar", "Greece", "Guernsey", "Holy_See", "Hungary", "Iceland", "Ireland",
               "Isle_of_Man", "Italy", "Jersey", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
               "Moldova", "Monaco", "Montenegro", "Netherlands", "North_Macedonia", "Norway", "Poland", "Portugal",
               "Romania", "Russia", "San_Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
               "Turkey", "Ukraine", "United_Kingdom"]
TO_OWID = {c: c.replace("_", " ") for c in ECDC_EUROPE}
TO_OWID["Holy_See"] = "Vatican"
FROM_OWID = {v: k for k, v in TO_OWID.items()}

DAILY_COLS = ["location", "date", "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million",
              "total_tests_per_thousand", "new_tests_smoothed_per_thousand", "positive_rate", "tests_per_case",
              "tests_units", "stringency_index", "population"]


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return pd.read_csv(io.BytesIO(r.read()), low_memory=False)


def main():
    daily = fetch(URL_DAILY)
    daily = daily[daily["location"].isin(FROM_OWID) & daily["date"].between("2020-01-01", "2020-12-31")][DAILY_COLS]
    daily.insert(0, "countriesAndTerritories", daily["location"].map(FROM_OWID))
    daily.to_csv(OUT / "owid_daily_europe_2020.csv", index=False)

    ex = fetch(URL_EXCESS)
    ex = ex[ex["location"].isin(FROM_OWID) & ex["date"].between("2020-01-01", "2021-01-10")]
    ex = ex[["location", "date", "time", "time_unit", "p_scores_all_ages", "p_proj_all_ages",
             "excess_proj_all_ages", "excess_per_million_proj_all_ages", "cum_excess_per_million_proj_all_ages"]]
    ex.insert(0, "countriesAndTerritories", ex["location"].map(FROM_OWID))
    ex.to_csv(OUT / "owid_excess_mortality_europe_2020.csv", index=False)
    print("daily:", daily.shape, daily["countriesAndTerritories"].nunique(), "countries")
    print("excess:", ex.shape, ex["countriesAndTerritories"].nunique(), "countries",
          ex.groupby("time_unit")["countriesAndTerritories"].nunique().to_dict())


if __name__ == "__main__":
    main()
