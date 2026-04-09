from dagster import Definitions, load_assets_from_modules, load_asset_checks_from_modules
from . import assets, sensors, schedules

all_assets = load_assets_from_modules([assets])
all_checks = load_asset_checks_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    schedules=[schedules.weekly_arxiv_schedule],
    sensors=[sensors.hot_folder_sensor],
)
