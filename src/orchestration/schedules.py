from dagster import ScheduleDefinition, AssetSelection

# --- CONFIGURATION ---
# Chạy lúc 23:00 Chủ Nhật hàng tuần (Cron: 0 23 * * 0)
WEEKLY_SCHEDULE_CRON = "0 23 * * 0"

weekly_arxiv_schedule = ScheduleDefinition(
    name="weekly_arxiv_full_sync",
    cron_schedule=WEEKLY_SCHEDULE_CRON,
    target=AssetSelection.all(),
    execution_timezone="Asia/Ho_Chi_Minh" # Cấu hình múi giờ phù hợp cho người dùng
)
