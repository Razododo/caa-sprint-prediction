"""
采集田协数据库中指定径赛项目的运动员资料（男女）。

当前默认项目：
200米, 400米, 800米, 1500米, 3000米, 5000米, 10000米

输出：
- data/raw/*.csv（与现有100m格式一致）
- data/raw/.progress_track_events.txt（增量进度）
"""
import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_URL = "https://athletics.fairplaycloud.com/scoreapi"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw"
PROGRESS_FILE = OUTPUT_DIR / ".progress_track_events.txt"

TARGET_EVENTS = {"200米", "400米", "800米", "1500米", "3000米", "5000米", "10000米"}

REQUEST_DELAY = 0.30
MAX_RETRIES = 3

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def api_get(path: str, params: dict | None = None):
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://athletics.fairplaycloud.com/",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
                data = json.loads(resp.read())
            time.sleep(REQUEST_DELAY)
            return data
        except Exception as exc:
            print(f"  [GET重试 {attempt + 1}/{MAX_RETRIES}] {path} 失败: {exc}")
            time.sleep(2 * (attempt + 1))
    return None


def api_post(path: str, body: dict):
    url = BASE_URL + path
    payload = json.dumps(body).encode("utf-8")
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Content-Type": "application/json;charset=UTF-8",
                    "Referer": "https://athletics.fairplaycloud.com/",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
                data = json.loads(resp.read())
            time.sleep(REQUEST_DELAY)
            return data
        except Exception as exc:
            print(f"  [POST重试 {attempt + 1}/{MAX_RETRIES}] {path} 失败: {exc}")
            time.sleep(2 * (attempt + 1))
    return None


def ts_to_date(ts_ms):
    if not ts_ms:
        return ""
    try:
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
    except Exception:
        return ""


def format_wind_suffix(rec: dict) -> str:
    if rec.get("wind") != 1:
        return ""
    ws = rec.get("windSpeed")
    if ws is None or ws >= 10000 or ws <= -10000:
        return ""
    try:
        val = ws / 100.0
        sign = "+" if val >= 0 else ""
        return f" ({sign}{val:.2f})"
    except Exception:
        return ""


def get_games_for_year(year: int) -> list[dict]:
    resp = api_post("/api/sportermeeting/list", {"year": str(year), "pageNo": 1, "pageSize": 500})
    if not resp or resp.get("code") != 0:
        return []
    return resp.get("data", [])


def build_years(start_year: int, end_year: int) -> list[int]:
    # 从新到旧抓取，便于尽快拿到近年数据并持续增量
    return list(range(end_year, start_year - 1, -1))


def get_events_for_game(game_id: str) -> list[dict]:
    resp = api_get(f"/api/sexsorteventlayer/field/{game_id}")
    if not resp or resp.get("code") != 0:
        return []
    events = []
    for field_group in resp.get("data", []):
        for item in (field_group.get("list") or []):
            events.append(item)
    return events


def filter_target_events(events: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for ev in events:
        event_name = ev.get("event", "")
        if event_name not in TARGET_EVENTS:
            continue
        key = (ev.get("sexSortEventID"), ev.get("layer", 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def get_athletes_for_game_by_events(game_id: str, event_ids: list[str]) -> list[dict]:
    # 对目标项目逐个 event_id 请求，再做 sporterID 去重，避免“锚点项目”漏抓
    seen_ids = set()
    athletes = []
    for event_id in event_ids:
        resp = api_post(
            "/api/sporter/findEventSporter",
            {
                "gameId": game_id,
                "sexSortEventID": event_id,
            },
        )
        if not resp or resp.get("code") != 0:
            continue
        for event_data in resp.get("data", []):
            if event_data.get("event") not in TARGET_EVENTS:
                continue
            for sporter in event_data.get("sporterList", []):
                sid = str(sporter.get("sporterID", ""))
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                athletes.append(sporter)
    return athletes


def get_athlete_details(sporter_id: str):
    resp = api_get("/api/sporter/findInfoById", {"sporterId": sporter_id})
    if not resp or resp.get("code") != 0:
        return None
    return resp.get("data")


def build_filename(athlete_data: dict) -> str:
    name = (athlete_data.get("familyNameCHN", "") + athlete_data.get("givenNameCHN", "")).strip()
    if not name:
        name = athlete_data.get("familyNameENG", "Unknown")
    birth = ts_to_date(athlete_data.get("birth"))
    gender = "男" if athlete_data.get("sex") == 0 else "女"

    safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_birth = birth.replace("-", "") if birth else ""
    if safe_birth:
        return f"{safe_name}_{safe_birth}_{gender}.csv"
    return f"{safe_name}_{gender}.csv"


def save_athlete_csv(athlete_data: dict, output_dir: Path) -> Path:
    name = (athlete_data.get("familyNameCHN", "") + athlete_data.get("givenNameCHN", "")).strip()
    if not name:
        name = athlete_data.get("familyNameENG", "Unknown")
    birth = ts_to_date(athlete_data.get("birth"))
    sex = "男" if athlete_data.get("sex") == 0 else "女"

    filepath = output_dir / build_filename(athlete_data)
    with filepath.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["运动员详情"])
        writer.writerow(["姓名", name])
        writer.writerow(["出生日期", birth])

        all_units = set()
        for records in (athlete_data.get("eventResultMap") or {}).values():
            for rec in records or []:
                unit_name = rec.get("unitNameCHN", "")
                if unit_name:
                    all_units.add(unit_name)
        writer.writerow(
            ["代表队(本次入口)", athlete_data.get("unitCHN", "")]
            if not all_units
            else ["代表队(历史)", " / ".join(sorted(all_units))]
        )
        writer.writerow(["性别", sex])
        writer.writerow(["注册号", athlete_data.get("registerNo", "")])
        writer.writerow(["身高(cm)", athlete_data.get("height", "")])
        writer.writerow(["体重(kg)", athlete_data.get("weight", "")])
        writer.writerow([])

        writer.writerow(["个人荣誉 - 历史最好成绩"])
        writer.writerow(["项目", "赛事名称", "项目全称", "名次", "成绩/得分", "比赛时间", "当时代表队", "注册单位"])
        event_result_map = athlete_data.get("eventResultMap") or {}
        for event_name, records in event_result_map.items():
            if not records:
                continue
            for rec in records:
                result_cell = (rec.get("showResult", "") or "") + format_wind_suffix(rec)
                writer.writerow(
                    [
                        event_name,
                        rec.get("gameNameCHN", ""),
                        rec.get("sexSortEvent", "") + rec.get("layerName", ""),
                        rec.get("splace", ""),
                        result_cell,
                        ts_to_date(rec.get("gameTime")),
                        rec.get("unitNameCHN", ""),
                        rec.get("registerUnitName", ""),
                    ]
                )

        writer.writerow([])
        writer.writerow(["个人最好成绩 (PB)"])
        writer.writerow(["项目", "赛事名称", "名次", "成绩/得分", "比赛时间", "当时代表队", "注册单位"])
        for pb in athlete_data.get("pb") or []:
            pb_result = (pb.get("showResult", "") or "") + format_wind_suffix(pb)
            writer.writerow(
                [
                    pb.get("sexSortEvent", ""),
                    pb.get("gameNameCHN", ""),
                    pb.get("splace", ""),
                    pb_result,
                    ts_to_date(pb.get("gameTime")),
                    pb.get("unitNameCHN", ""),
                    pb.get("registerUnitName", ""),
                ]
            )
    return filepath


def scan_existing_keys(output_dir: Path) -> set[str]:
    keys = set()
    for fp in output_dir.glob("*.csv"):
        stem = fp.stem
        if stem.endswith("_男") or stem.endswith("_女"):
            keys.add(stem)
    return keys


def load_progress(progress_file: Path):
    sporter_ids, register_nos, name_birth_keys = set(), set(), set()
    if not progress_file.exists():
        return sporter_ids, register_nos, name_birth_keys
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                sporter_ids.add(parts[0])
            if len(parts) > 1 and parts[1]:
                register_nos.add(parts[1])
            if len(parts) > 2 and parts[2]:
                name_birth_keys.add(parts[2])
    return sporter_ids, register_nos, name_birth_keys


def save_progress(progress_file: Path, sporter_id: str, register_no: str = "", key: str = ""):
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"{sporter_id}\t{register_no}\t{key}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1990, help="历史起始年份（含）")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="结束年份（含）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    collected_ids, collected_regnos, collected_keys = load_progress(PROGRESS_FILE)
    collected_keys |= scan_existing_keys(OUTPUT_DIR)

    print("目标项目:", ", ".join(sorted(TARGET_EVENTS)))
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"已有进度: sporterID={len(collected_ids)}, registerNo={len(collected_regnos)}, name_key={len(collected_keys)}")

    years = build_years(args.start_year, args.end_year)
    print(f"采集年份范围: {args.start_year}-{args.end_year}（共{len(years)}年）")

    total_new, skipped_dup = 0, 0
    for year in years:
        games = get_games_for_year(year)
        print(f"\n==== 年份 {year}，比赛数: {len(games)} ====", flush=True)
        for idx, game in enumerate(games, start=1):
            game_id = game.get("gameID")
            game_name = game.get("gameNameCHN", str(game_id))
            print(f"\n[{idx}/{len(games)}] {game_name}", flush=True)

            events = get_events_for_game(str(game_id))
            target_events = filter_target_events(events)
            if not target_events:
                print("  无目标项目，跳过", flush=True)
                continue

            event_ids = [str(ev.get("sexSortEventID")) for ev in target_events if ev.get("sexSortEventID")]
            athletes = get_athletes_for_game_by_events(str(game_id), event_ids)
            print(f"  目标项目去重后运动员数: {len(athletes)}", flush=True)

            for ath in athletes:
                sid = str(ath.get("sporterID", ""))
                if not sid or sid in collected_ids:
                    continue

                ath_name = (ath.get("familyNameCHN", "") + ath.get("givenNameCHN", "")).strip()
                ath_birth = ts_to_date(ath.get("birth", 0) or 0).replace("-", "") if ath.get("birth") else ""
                ath_gender = "男" if ath.get("sex") == 0 else "女"
                key = f"{ath_name}_{ath_birth}_{ath_gender}"
                if key in collected_keys:
                    collected_ids.add(sid)
                    skipped_dup += 1
                    continue

                detail = get_athlete_details(sid)
                if not detail:
                    continue
                reg_no = detail.get("registerNo", "")
                if reg_no and reg_no in collected_regnos:
                    collected_ids.add(sid)
                    collected_keys.add(key)
                    save_progress(PROGRESS_FILE, sid, reg_no, key)
                    skipped_dup += 1
                    continue

                try:
                    out_fp = save_athlete_csv(detail, OUTPUT_DIR)
                    file_key = out_fp.stem
                    collected_ids.add(sid)
                    if reg_no:
                        collected_regnos.add(reg_no)
                    collected_keys.add(file_key)
                    save_progress(PROGRESS_FILE, sid, reg_no, file_key)
                    total_new += 1
                    print(f"    采集: {ath_name} -> {out_fp.name}", flush=True)
                except Exception as exc:
                    print(f"    保存失败: {ath_name} ({sid}) - {exc}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"采集完成：新增 {total_new}，跳过重复 {skipped_dup}", flush=True)
    print(f"累计 name_key: {len(collected_keys)}, registerNo: {len(collected_regnos)}", flush=True)
    print(f"数据目录: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
