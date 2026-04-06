from dagster import Definitions, load_assets_from_modules
from . import assets, sensors, schedules

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    schedules=[schedules.weekly_arxiv_schedule],
    sensors=[sensors.hot_folder_sensor],
)
