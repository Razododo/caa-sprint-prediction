"""
Step 3: Map team names → province-level units.

Input:  data/interim/cleaned_athletes.parquet
Output: Updated province / max_team_level columns in cleaned_athletes.parquet
        data/interim/province_mapping.csv (unique team → province mapping)
        data/interim/unmapped_teams.csv (teams that couldn't be auto-mapped)
"""
import logging
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import yaml

from utils.io import load_config, ensure_dirs, read_parquet, write_parquet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CITY_TO_PROVINCE = {
    "北京": "北京", "天津": "天津", "上海": "上海", "重庆": "重庆",
    "石家庄": "河北", "唐山": "河北", "秦皇岛": "河北", "邯郸": "河北", "保定": "河北", "廊坊": "河北",
    "太原": "山西", "大同": "山西", "运城": "山西", "临汾": "山西",
    "呼和浩特": "内蒙古", "包头": "内蒙古", "鄂尔多斯": "内蒙古", "赤峰": "内蒙古",
    "沈阳": "辽宁", "大连": "辽宁", "鞍山": "辽宁", "抚顺": "辽宁", "锦州": "辽宁", "丹东": "辽宁",
    "长春": "吉林", "吉林市": "吉林", "延边": "吉林", "四平": "吉林",
    "哈尔滨": "黑龙江", "齐齐哈尔": "黑龙江", "大庆": "黑龙江", "牡丹江": "黑龙江",
    "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "常州": "江苏", "徐州": "江苏",
    "杭州": "浙江", "宁波": "浙江", "温州": "浙江", "绍兴": "浙江", "金华": "浙江", "台州": "浙江",
    "合肥": "安徽", "芜湖": "安徽", "蚌埠": "安徽", "阜阳": "安徽",
    "福州": "福建", "厦门": "福建", "泉州": "福建", "漳州": "福建", "莆田": "福建",
    "南昌": "江西", "赣州": "江西", "九江": "江西",
    "济南": "山东", "青岛": "山东", "烟台": "山东", "潍坊": "山东", "淄博": "山东", "威海": "山东",
    "郑州": "河南", "洛阳": "河南", "开封": "河南", "新乡": "河南", "许昌": "河南", "南阳": "河南",
    "武汉": "湖北", "宜昌": "湖北", "襄阳": "湖北", "荆州": "湖北",
    "长沙": "湖南", "株洲": "湖南", "湘潭": "湖南", "衡阳": "湖南", "常德": "湖南",
    "广州": "广东", "深圳": "广东", "珠海": "广东", "东莞": "广东", "佛山": "广东", "中山": "广东", "惠州": "广东",
    "南宁": "广西", "柳州": "广西", "桂林": "广西", "梧州": "广西", "北海": "广西",
    "海口": "海南", "三亚": "海南",
    "成都": "四川", "绵阳": "四川", "德阳": "四川", "宜宾": "四川", "自贡": "四川",
    "贵阳": "贵州", "遵义": "贵州", "六盘水": "贵州",
    "昆明": "云南", "曲靖": "云南", "大理": "云南",
    "拉萨": "西藏",
    "西安": "陕西", "咸阳": "陕西", "宝鸡": "陕西", "渭南": "陕西",
    "兰州": "甘肃", "天水": "甘肃", "酒泉": "甘肃",
    "西宁": "青海",
    "银川": "宁夏",
    "乌鲁木齐": "新疆", "喀什": "新疆", "伊犁": "新疆",
    "香港": "香港", "澳门": "澳门",
}

NATIONAL_KEYWORDS = {"国家队", "CHN", "中国", "China"}
INDEPENDENT_KEYWORDS = {"个人组", "大众组"}


def map_team_to_province(team: str, direct_provinces: set[str]) -> str | None:
    """Map a single team name to a province. Returns None if unmapped."""
    team = team.strip()
    if not team:
        return None

    if team in NATIONAL_KEYWORDS or team in INDEPENDENT_KEYWORDS:
        return None

    # Direct province name match
    for p in direct_provinces:
        if team == p or team.startswith(p):
            return p

    # City-to-province lookup
    for city, prov in CITY_TO_PROVINCE.items():
        if city in team:
            return prov

    # Province name appears anywhere in team string
    for p in direct_provinces:
        if p in team:
            return p

    return None


def classify_team_level(team: str) -> str:
    """Classify a team into a hierarchy level."""
    team = team.strip()
    if team in NATIONAL_KEYWORDS:
        return "national"
    if any(k in team for k in ["省", "自治区"]):
        return "provincial"
    for p in CITY_TO_PROVINCE.values():
        if p in team:
            return "provincial"
    if any(k in team for k in ["市", "区", "县", "州"]):
        return "municipal"
    if any(k in team for k in ["大学", "学院", "中学", "学校", "体校"]):
        return "school"
    return "other"


def get_max_level(levels: list[str]) -> str:
    """Return highest level from a list."""
    order = {"national": 4, "provincial": 3, "municipal": 2, "school": 1, "other": 0}
    if not levels:
        return "other"
    return max(levels, key=lambda x: order.get(x, 0))


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    interim = Path(config["paths"]["interim"])
    df_ath = read_parquet(interim / "cleaned_athletes.parquet")

    direct_provinces = set(config["province"]["direct_names"])

    all_mappings: list[dict] = []
    provinces = []
    max_levels = []

    for _, row in df_ath.iterrows():
        teams_raw = row.get("team_history", "")
        if isinstance(teams_raw, str):
            teams = [t.strip() for t in teams_raw.split("/") if t.strip()]
        else:
            teams = []

        mapped_provs = []
        team_levels = []

        for t in teams:
            prov = map_team_to_province(t, direct_provinces)
            if prov:
                mapped_provs.append(prov)
            level = classify_team_level(t)
            team_levels.append(level)
            all_mappings.append({"team": t, "province": prov, "level": level})

        home_prov = mapped_provs[0] if mapped_provs else None
        ml = get_max_level(team_levels)
        provinces.append(home_prov)
        max_levels.append(ml)

    df_ath["province"] = provinces
    df_ath["max_team_level"] = max_levels

    write_parquet(df_ath, interim / "cleaned_athletes.parquet")

    # Province mapping table (unique teams)
    df_map = pd.DataFrame(all_mappings).drop_duplicates(subset=["team"])
    df_map.to_csv(interim / "province_mapping.csv", index=False, encoding="utf-8-sig")

    unmapped = df_map[df_map["province"].isna()]
    unmapped.to_csv(interim / "unmapped_teams.csv", index=False, encoding="utf-8-sig")

    # Summary
    mapped_count = df_ath["province"].notna().sum()
    total = len(df_ath)
    logger.info(f"Province mapped: {mapped_count}/{total} ({mapped_count/total*100:.1f}%)")
    logger.info(f"Unique teams: {len(df_map)}, unmapped: {len(unmapped)}")
    logger.info(f"Province distribution (top 15):")
    prov_counts = df_ath["province"].value_counts().head(15)
    for p, c in prov_counts.items():
        logger.info(f"  {p}: {c}")
    logger.info(f"Team level distribution:")
    for lv, c in df_ath["max_team_level"].value_counts().items():
        logger.info(f"  {lv}: {c}")


if __name__ == "__main__":
    main()
