import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np
import altair as alt

from pathlib import Path
from matplotlib import colormaps
from matplotlib.colors import Normalize

# 1. 페이지 설정
st.set_page_config(
    page_title="1000에50, 우리집을 찾아서",
    page_icon="🔍",
    layout="wide"
)

st.title("서울시 1000에 50 지도, 우리집을 찾아서🔍")
st.caption(
    "서울시 전월세 실거래 데이터를 바탕으로 조건에 맞는 "
    "월세 수준을 법정동·500m 격자·지하철역별로 비교하고, "
    "LH 청년매입임대 공급주택을 함께 살펴봅니다."
)

# 2. 데이터 경로
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 3. 데이터 로드
@st.cache_data(show_spinner=False)
def load_rent(year):
    return pd.read_parquet(
        DATA_DIR / f"seoul_rent_{year}.parquet"
    )

@st.cache_data(show_spinner=False)
def load_dong():
    gdf = gpd.read_file(
        DATA_DIR / "seoul_dong_boundary.geojson"
    )
    return gdf.to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_grid():
    gdf = gpd.read_file(
        DATA_DIR / "seoul_500m_grid.geojson"
    )
    return gdf.to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_gu():
    gdf = gpd.read_file(
        DATA_DIR / "seoul_gu.geojson"
    )
    return gdf.to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_subway():
    gdf = gpd.read_file(
        DATA_DIR / "subway.geojson"
    )
    return gdf.to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_rent_station():
    return pd.read_parquet(
        DATA_DIR / "rent_station.parquet"
    )

@st.cache_data(show_spinner=False)
def load_policy_units():
    return pd.read_parquet(
        DATA_DIR / "policy_housing_units.parquet"
    )

@st.cache_data(show_spinner=False)
def load_policy_buildings():
    return pd.read_parquet(
        DATA_DIR / "policy_housing_buildings.parquet"
    )

# 4. 기본 설정
MIN_MAP_COUNT = 5
MIN_DONG_LABEL_COUNT = 20

subway_reference = load_subway()

all_subway_lines = sorted({
    line.strip()
    for text in subway_reference["hoseon"].dropna()
    for line in str(text).split(",")
})

DEPOSIT_BASE_VALUES = [
    0, 500, 1000, 2000, 3000, 4000,
    5000, 6000, 7000, 8000, 9000, 10000
]

def deposit_label(value):
    if value == 0:
        return "무보증"
    return f"{value:,}만원"

DEPOSIT_OPTIONS = {
    deposit_label(value): value
    for value in DEPOSIT_BASE_VALUES
}

def get_deposit_range(base_deposit):
    if base_deposit == 0:
        return 0, 1
    if base_deposit == 500:
        return 1, 1000
    return base_deposit, base_deposit + 1000

# 5. 위젯 기본값
st.session_state.setdefault("house_type_widget", "전체")
st.session_state.setdefault("spatial_unit_widget", "법정동별")
st.session_state.setdefault("subway_lines_widget", [])
st.session_state.setdefault("year_widget", 2025)
st.session_state.setdefault("deposit_widget", "1,000만원")
st.session_state.setdefault("area_widget", (15, 30))
st.session_state.setdefault("age_widget", (0, 100))
st.session_state.setdefault("floor_widget", "지하·반지하 제외")
st.session_state.setdefault("count_label_widget", False)
st.session_state.setdefault("show_policy_widget", True)
st.session_state.setdefault("policy_priority_widget", "청년 1순위")

# 6. 프리셋
def apply_preset(
    deposit,
    area,
    age,
    floor="지하·반지하 제외",
    house_type="전체"
):
    st.session_state["house_type_widget"] = house_type
    st.session_state["spatial_unit_widget"] = "법정동별"
    st.session_state["subway_lines_widget"] = []
    st.session_state["year_widget"] = 2025
    st.session_state["deposit_widget"] = deposit
    st.session_state["area_widget"] = area
    st.session_state["age_widget"] = age
    st.session_state["floor_widget"] = floor
    st.session_state["show_policy_widget"] = True
    st.session_state["policy_priority_widget"] = "청년 1순위"
    st.session_state["preset_run"] = True

def reset_filters():
    st.session_state["house_type_widget"] = "전체"
    st.session_state["spatial_unit_widget"] = "법정동별"
    st.session_state["subway_lines_widget"] = []
    st.session_state["year_widget"] = 2025
    st.session_state["deposit_widget"] = "1,000만원"
    st.session_state["area_widget"] = (15, 30)
    st.session_state["age_widget"] = (0, 100)
    st.session_state["floor_widget"] = "지하·반지하 제외"
    st.session_state["count_label_widget"] = False
    st.session_state["show_policy_widget"] = True
    st.session_state["policy_priority_widget"] = "청년 1순위"
    st.session_state["preset_run"] = False

# 7. 검색 조건
with st.sidebar.form("search_form"):
    st.header("🔍 검색 조건 설정")

    house_type_selection = st.radio(
        "주택 유형",
        ["전체", "연립다세대", "오피스텔"],
        key="house_type_widget"
    )

    spatial_unit = st.radio(
        "시각화 단위",
        ["법정동별", "격자별", "지하철역별"],
        key="spatial_unit_widget"
    )

    selected_subway_lines = st.multiselect(
        "지하철 노선",
        options=all_subway_lines,
        key="subway_lines_widget",
        help=(
            "지하철역별 시각화에 적용됩니다. "
            "선택하지 않으면 전체 역을 표시합니다."
        )
    )

    selected_year = st.selectbox(
        "연도",
        [2025, 2024, 2023],
        key="year_widget"
    )

    selected_deposit_label = st.selectbox(
        "기준 보증금",
        list(DEPOSIT_OPTIONS.keys()),
        key="deposit_widget",
        help=(
            "선택한 보증금 구간의 거래를 대표 보증금으로 "
            "환산하여 비교합니다."
        )
    )

    area_min, area_max = st.slider(
        "전용면적 (㎡)",
        min_value=5,
        max_value=85,
        step=1,
        key="area_widget"
    )

    st.caption(
        f"{area_min}㎡ - {area_max}㎡ "
        f"(약 {area_min / 3.3058:.1f}평 - "
        f"{area_max / 3.3058:.1f}평)"
    )

    age_min, age_max = st.slider(
        "건물 연식 (년)",
        min_value=0,
        max_value=100,
        step=1,
        key="age_widget"
    )

    st.caption(
        f"준공 후 {age_min}년 - {age_max}년 "
        f"(약 {selected_year - age_max}년 - "
        f"{selected_year - age_min}년 준공)"
    )

    selected_floor = st.selectbox(
        "층수",
        [
            "전체",
            "지하·반지하 제외",
            "저층 (1층 이하)"
        ],
        key="floor_widget"
    )

    show_count_labels = st.checkbox(
        "법정동 거래건수 지도에 표시",
        key="count_label_widget",
        help=(
            "법정동별 지도에서 일정 거래건수 이상의 "
            "지역에 거래건수를 직접 표시합니다."
        )
    )

    st.markdown("##### 🏠 정책주택")

    show_policy = st.checkbox(
        "LH 청년매입임대 함께 보기",
        key="show_policy_widget",
        help=(
            "선택한 면적·층수·주택유형 조건에 맞는 "
            "LH 청년매입임대 공급주택을 지도에 표시합니다."
        )
    )

    policy_priority = st.selectbox(
        "정책주택 임대조건",
        ["청년 1순위", "청년 2·3순위"],
        key="policy_priority_widget",
        disabled=not show_policy
    )

    submit_button = st.form_submit_button(
        "우리집 찾기",
        type="primary",
        use_container_width=True
    )

# 8. 빠른 검색
st.sidebar.markdown("### 🏠 빠른 검색")
st.sidebar.caption(
    "자주 찾는 원룸 조건을 한 번에 적용합니다."
)

preset_col1, preset_col2 = st.sidebar.columns(2)

preset_col1.button(
    "1000 · 4-5평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "1,000만원",
        (13, 17),
        (0, 25)
    )
)

preset_col2.button(
    "1000 · 5-7평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "1,000만원",
        (16, 23),
        (0, 25)
    )
)

preset_col3, preset_col4 = st.sidebar.columns(2)

preset_col3.button(
    "2000 · 신축원룸",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "2,000만원",
        (16, 23),
        (0, 10)
    )
)

preset_col4.button(
    "조건 초기화",
    use_container_width=True,
    on_click=reset_filters
)

# 프리셋 버튼도 검색 실행
run_search = (
    submit_button
    or st.session_state.pop("preset_run", False)
)

# 9. 공통 조건 필터
def filter_common_data(
    df,
    house_type,
    contract_year,
    age_min,
    age_max,
    floor
):
    df = df[
        df["전월세구분"] == "월세"
    ].copy()

    if house_type != "전체":
        df = df[
            df["건물용도"] == house_type
        ]

    df["건물연식"] = (
        contract_year - df["건축년도"]
    )

    df = df[
        (df["건물연식"] >= age_min) &
        (df["건물연식"] <= age_max)
    ]

    if floor == "지하·반지하 제외":
        df = df[
            df["층"] > 0
        ]

    elif floor == "저층 (1층 이하)":
        df = df[
            df["층"] <= 1
        ]

    return df

# 10. 지도용 거래 필터
def filter_rent_data(
    df,
    dep_min,
    dep_max,
    base_deposit,
    area_min,
    area_max
):
    df = df[
        (df["보증금(만원)"] >= dep_min) &
        (df["보증금(만원)"] < dep_max) &
        (df["임대면적"] >= area_min) &
        (df["임대면적"] <= area_max)
    ].copy()

    df["환산월세(만원)"] = (
        df["임대료(만원)"]
        + (df["보증금(만원)"] - base_deposit) * 0.005
    )

    return df

# 11. 공간 단위별 통계
def aggregate_rent(df, group_cols):
    stats = (
        df.groupby(group_cols)["환산월세(만원)"]
        .agg(
            거래건수="count",
            평균="mean",
            중앙="median",
            최저="min",
            최고="max"
        )
        .reset_index()
    )

    for col in ["평균", "중앙", "최저", "최고"]:
        stats[col] = stats[col].round(1)

    return stats

# 12. 지도 색상
def add_map_color(data, value_col="평균"):
    valid_values = data[value_col].dropna()

    vmin = valid_values.quantile(0.05)
    vmax = valid_values.quantile(0.95)

    if vmin == vmax:
        vmax = vmin + 1

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=True
    )

    cmap = colormaps["RdYlBu_r"]

    def make_color(value):
        if pd.isna(value):
            return [220, 220, 220, 0]

        rgba = cmap(norm(value))

        return [
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
            190
        ]

    data = data.copy()
    data["fill_color"] = data[value_col].apply(
        make_color
    )

    return data, vmin, vmax

# 13. 법정동 거래건수 라벨
def make_dong_label_data(
    dong_map,
    min_count=MIN_DONG_LABEL_COUNT
):
    label_gdf = dong_map[
        dong_map["거래건수"] >= min_count
    ][
        [
            "EMD_NM",
            "거래건수",
            "geometry"
        ]
    ].copy()

    if label_gdf.empty:
        return pd.DataFrame()

    # 대표 위치 계산은 투영좌표계에서 수행
    label_gdf = label_gdf.to_crs(epsg=5179)
    label_gdf["geometry"] = (
        label_gdf.geometry.representative_point()
    )
    label_gdf = label_gdf.to_crs(epsg=4326)

    label_gdf["longitude"] = (
        label_gdf.geometry.x
    )
    label_gdf["latitude"] = (
        label_gdf.geometry.y
    )

    label_gdf["label"] = (
        label_gdf["거래건수"]
        .astype(int)
        .astype(str)
        + "건"
    )

    return pd.DataFrame(
        label_gdf[
            [
                "EMD_NM",
                "거래건수",
                "longitude",
                "latitude",
                "label"
            ]
        ]
    )

# 14. 보증금별 월세 통계
def make_deposit_stats(
    df,
    area_min,
    area_max
):
    chart_df = df[
        (df["임대면적"] >= area_min) &
        (df["임대면적"] <= area_max)
    ].copy()

    chart_df["보증금기준"] = pd.NA

    chart_df.loc[
        chart_df["보증금(만원)"] == 0,
        "보증금기준"
    ] = "무보증"

    chart_df.loc[
        (chart_df["보증금(만원)"] >= 1) &
        (chart_df["보증금(만원)"] < 1000),
        "보증금기준"
    ] = "500만원"

    for base in range(1000, 11000, 1000):
        chart_df.loc[
            (chart_df["보증금(만원)"] >= base) &
            (chart_df["보증금(만원)"] < base + 1000),
            "보증금기준"
        ] = f"{base:,}만원"

    chart_df["보증금기준"] = pd.Categorical(
        chart_df["보증금기준"],
        categories=list(DEPOSIT_OPTIONS.keys()),
        ordered=True
    )

    stats = (
        chart_df.dropna(
            subset=["보증금기준"]
        )
        .groupby(
            "보증금기준",
            observed=True
        )
        .agg(
            거래건수=("임대료(만원)", "count"),
            평균월세=("임대료(만원)", "mean"),
            중앙월세=("임대료(만원)", "median")
        )
        .reset_index()
    )

    stats["평균월세"] = (
        stats["평균월세"].round(1)
    )
    stats["중앙월세"] = (
        stats["중앙월세"].round(1)
    )

    return stats

# 15. 면적별 월세 통계
def make_area_stats(
    df,
    dep_min,
    dep_max,
    base_deposit
):
    chart_df = df[
        (df["보증금(만원)"] >= dep_min) &
        (df["보증금(만원)"] < dep_max) &
        (df["임대면적"] >= 5) &
        (df["임대면적"] <= 85)
    ].copy()

    chart_df["환산월세(만원)"] = (
        chart_df["임대료(만원)"]
        + (
            chart_df["보증금(만원)"]
            - base_deposit
        ) * 0.005
    )

    bins = list(range(5, 95, 5))

    labels = [
        f"{start}-{start + 5}"
        for start in bins[:-1]
    ]

    chart_df["면적대"] = pd.cut(
        chart_df["임대면적"],
        bins=bins,
        labels=labels,
        right=False
    )

    stats = (
        chart_df.dropna(
            subset=["면적대"]
        )
        .groupby(
            "면적대",
            observed=True
        )
        .agg(
            거래건수=("환산월세(만원)", "count"),
            평균월세=("환산월세(만원)", "mean"),
            중앙월세=("환산월세(만원)", "median")
        )
        .reset_index()
    )

    stats["평균월세"] = (
        stats["평균월세"].round(1)
    )
    stats["중앙월세"] = (
        stats["중앙월세"].round(1)
    )

    return stats

# 16. 정책주택 조건 필터 및 공식 임대조건 비교
def prepare_policy_data(
    policy_units,
    policy_buildings,
    house_type,
    area_min,
    area_max,
    floor,
    priority,
    base_deposit
):
    units = policy_units.copy()

    # 민간 검색조건 중 정책주택 자료로 확인 가능한 항목만 적용
    units = units[
        (units["전용면적"] >= area_min) &
        (units["전용면적"] <= area_max)
    ].copy()

    if house_type == "오피스텔":
        units = units[
            units["주택유형"] == "오피스텔"
        ].copy()

    elif house_type == "연립다세대":
        units = units[
            units["주택유형"]
            .astype(str)
            .str.contains(
                "다세대|연립",
                regex=True,
                na=False
            )
        ].copy()

    if floor == "지하·반지하 제외":
        units = units[
            units["층"] > 0
        ].copy()

    elif floor == "저층 (1층 이하)":
        units = units[
            units["층"] <= 1
        ].copy()

    if priority == "청년 1순위":
        prefix = "청년1순위"
    else:
        prefix = "청년23순위"

    basic_dep_col = f"{prefix}_기본보증금_만원"
    basic_rent_col = f"{prefix}_기본월세_만원"
    max_dep_col = f"{prefix}_최대전환보증금_만원"
    max_rent_col = f"{prefix}_최대전환월세_만원"

    compare_cols = [
        basic_dep_col,
        basic_rent_col,
        max_dep_col,
        max_rent_col
    ]

    for col in compare_cols:
        units[col] = pd.to_numeric(
            units[col],
            errors="coerce"
        )

    units["정책월세_만원"] = np.nan

    dep_gap = (
        units[max_dep_col]
        - units[basic_dep_col]
    )

    comparable = (
        units[basic_dep_col].notna() &
        units[basic_rent_col].notna() &
        units[max_dep_col].notna() &
        units[max_rent_col].notna() &
        (dep_gap > 0) &
        (base_deposit >= units[basic_dep_col]) &
        (base_deposit <= units[max_dep_col])
    )

    # LH 공고의 기본 임대조건과 최대전환 임대조건 사이를
    # 선형 보간하여 선택 보증금에 대응하는 월세를 계산
    units.loc[
        comparable,
        "정책월세_만원"
    ] = (
        units.loc[
            comparable,
            basic_rent_col
        ]
        + (
            base_deposit
            - units.loc[
                comparable,
                basic_dep_col
            ]
        )
        * (
            units.loc[
                comparable,
                max_rent_col
            ]
            - units.loc[
                comparable,
                basic_rent_col
            ]
        )
        / dep_gap.loc[comparable]
    )

    units["정책월세_만원"] = (
        units["정책월세_만원"]
        .round(1)
    )

    units["비교가능"] = comparable

    if units.empty:
        return units, pd.DataFrame()

    policy_stats = (
        units.groupby(
            "building_id"
        )
        .agg(
            조건공급호수=(
                "policy_unit_id",
                "count"
            ),
            비교가능호수=(
                "비교가능",
                "sum"
            ),
            전용면적_최소=(
                "전용면적",
                "min"
            ),
            전용면적_최대=(
                "전용면적",
                "max"
            ),
            정책월세_평균=(
                "정책월세_만원",
                "mean"
            ),
            정책월세_중앙=(
                "정책월세_만원",
                "median"
            ),
            정책월세_최저=(
                "정책월세_만원",
                "min"
            ),
            정책월세_최고=(
                "정책월세_만원",
                "max"
            )
        )
        .reset_index()
    )

    for col in [
        "전용면적_최소",
        "전용면적_최대",
        "정책월세_평균",
        "정책월세_중앙",
        "정책월세_최저",
        "정책월세_최고"
    ]:
        policy_stats[col] = (
            policy_stats[col].round(1)
        )

    static_cols = [
        "building_id",
        "주택명",
        "지오코딩주소",
        "longitude",
        "latitude"
    ]

    policy_map = (
        policy_buildings[
            static_cols
        ]
        .merge(
            policy_stats,
            on="building_id",
            how="inner"
        )
    )

    policy_map = policy_map[
        policy_map["longitude"].notna() &
        policy_map["latitude"].notna()
    ].copy()

    policy_map["radius"] = (
        100
        + np.sqrt(
            policy_map["조건공급호수"]
        ) * 25
    ).clip(
        lower=120,
        upper=320
    )

    policy_map["tip_title"] = (
        policy_map["주택명"]
    )

    policy_map["tip_1"] = (
        "LH 청년매입임대 · "
        + policy_map[
            "조건공급호수"
        ].astype(int).astype(str)
        + "호"
    )

    policy_map["tip_2"] = (
        "전용 "
        + policy_map[
            "전용면적_최소"
        ].map(lambda x: f"{x:.1f}")
        + "-"
        + policy_map[
            "전용면적_최대"
        ].map(lambda x: f"{x:.1f}")
        + "㎡"
    )

    policy_map["tip_3"] = (
        priority
        + " · 보증금 "
        + f"{base_deposit:,}만원 기준"
    )

    policy_map["tip_4"] = policy_map[
        "정책월세_중앙"
    ].apply(
        lambda x:
            f"LH 전환월세 중앙 {x:.1f}만원"
            if pd.notna(x)
            else "선택 보증금은 공식 전환범위 밖"
    )

    policy_map["tip_5"] = (
        policy_map["지오코딩주소"]
    )

    return units, policy_map


def make_policy_layer(policy_map):
    if policy_map.empty:
        return None

    return pdk.Layer(
        "ScatterplotLayer",
        data=policy_map,
        get_position=[
            "longitude",
            "latitude"
        ],
        get_fill_color=[
            92, 76, 180, 220
        ],
        get_line_color=[
            255, 255, 255, 230
        ],
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=14,
        stroked=True,
        line_width_min_pixels=1.5,
        pickable=True,
        auto_highlight=True
    )


MAP_TOOLTIP = {
    "html": """
        <b>{tip_title}</b><br/>
        {tip_1}<br/>
        {tip_2}<br/>
        {tip_3}<br/>
        {tip_4}<br/>
        {tip_5}
    """
}

# 17. 서울 기본 지도
SEOUL_VIEW = pdk.ViewState(
    latitude=37.5665,
    longitude=126.9780,
    zoom=10.3,
    pitch=0
)

# 17. 검색 실행
if run_search:
    base_deposit = (
        DEPOSIT_OPTIONS[selected_deposit_label]
    )

    dep_min, dep_max = get_deposit_range(
        base_deposit
    )

    with st.spinner(
        f"{selected_year}년 거래 데이터를 불러오는 중..."
    ):
        df_raw = load_rent(selected_year)

    common_df = filter_common_data(
        df_raw,
        house_type_selection,
        selected_year,
        age_min,
        age_max,
        selected_floor
    )

    df = filter_rent_data(
        common_df,
        dep_min,
        dep_max,
        base_deposit,
        area_min,
        area_max
    )

    if df.empty:
        st.warning(
            "선택한 조건에 해당하는 거래가 없습니다."
        )
        st.stop()

    # 정책주택은 필요할 때만 불러오기
    policy_units_filtered = pd.DataFrame()
    policy_map = pd.DataFrame()
    policy_layer = None

    if show_policy:
        policy_units = load_policy_units()
        policy_buildings = load_policy_buildings()

        (
            policy_units_filtered,
            policy_map
        ) = prepare_policy_data(
            policy_units,
            policy_buildings,
            house_type_selection,
            area_min,
            area_max,
            selected_floor,
            policy_priority,
            base_deposit
        )

        policy_layer = make_policy_layer(
            policy_map
        )

    # 18. 결과 요약
    st.subheader(
        f"📊 {selected_year}년 "
        f"{house_type_selection} "
        f"{spatial_unit}"
    )

    st.markdown(
        f"**기준 보증금 {selected_deposit_label} · "
        f"전용면적 {area_min}-{area_max}㎡ · "
        f"건물 연식 {age_min}-{age_max}년**"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "거래건수",
        f"{len(df):,}건"
    )

    col2.metric(
        "평균 환산월세",
        f"{df['환산월세(만원)'].mean():.1f}만원"
    )

    col3.metric(
        "중앙 환산월세",
        f"{df['환산월세(만원)'].median():.1f}만원"
    )

    if base_deposit == 0:
        st.caption(
            "실제 보증금이 없는 월세 거래를 기준으로 비교합니다."
        )

    elif base_deposit == 500:
        st.caption(
            "실제 보증금 1-999만원 거래를 "
            "보증금 500만원 기준 월세로 환산했습니다."
        )

    else:
        st.caption(
            f"실제 보증금 {dep_min:,}-{dep_max - 1:,}만원 "
            f"거래를 보증금 {base_deposit:,}만원 기준 "
            f"월세로 환산했습니다."
        )

    # 19. 법정동별
    if spatial_unit == "법정동별":
        dong = load_dong()

        df["자치구코드"] = (
            df["자치구코드"].astype("Int64")
        )
        df["법정동코드"] = (
            df["법정동코드"].astype("Int64")
        )

        dong["자치구코드"] = (
            dong["자치구코드"].astype("Int64")
        )
        dong["법정동코드"] = (
            dong["법정동코드"].astype("Int64")
        )

        dong_stats = aggregate_rent(
            df,
            ["자치구코드", "법정동코드"]
        )

        dong_map = dong.merge(
            dong_stats,
            on=["자치구코드", "법정동코드"],
            how="inner"
        )

        dong_map = dong_map[
            dong_map["거래건수"] >= MIN_MAP_COUNT
        ].copy()

        if dong_map.empty:
            st.warning(
                "거래 5건 이상인 법정동이 없습니다."
            )
            st.stop()

        dong_map, vmin, vmax = add_map_color(
            dong_map
        )

        dong_map["tip_title"] = (
            dong_map["EMD_NM"]
        )
        dong_map["tip_1"] = (
            "거래건수 "
            + dong_map["거래건수"]
            .astype(int).astype(str)
            + "건"
        )
        dong_map["tip_2"] = (
            "평균 환산월세 "
            + dong_map["평균"]
            .map(lambda x: f"{x:.1f}만원")
        )
        dong_map["tip_3"] = (
            "중앙 환산월세 "
            + dong_map["중앙"]
            .map(lambda x: f"{x:.1f}만원")
        )
        dong_map["tip_4"] = (
            "최저 "
            + dong_map["최저"]
            .map(lambda x: f"{x:.1f}만원")
        )
        dong_map["tip_5"] = (
            "최고 "
            + dong_map["최고"]
            .map(lambda x: f"{x:.1f}만원")
        )

        dong_layer = pdk.Layer(
            "GeoJsonLayer",
            data=dong_map,
            filled=True,
            stroked=True,
            get_fill_color="fill_color",
            get_line_color=[90, 90, 90, 100],
            line_width_min_pixels=0.5,
            pickable=True,
            auto_highlight=True
        )

        layers = [dong_layer]

        if policy_layer is not None:
            layers.append(
                policy_layer
            )

        # 법정동 거래건수 라벨
        if show_count_labels:
            dong_label_data = make_dong_label_data(
                dong_map
            )

            if not dong_label_data.empty:
                count_text_layer = pdk.Layer(
                    "TextLayer",
                    data=dong_label_data,
                    get_position=[
                        "longitude",
                        "latitude"
                    ],
                    get_text="label",
                    get_size=11,
                    get_color=[40, 40, 40, 220],
                    get_text_anchor='"middle"',
                    get_alignment_baseline='"center"',
                    pickable=False
                )

                layers.append(
                    count_text_layer
                )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=SEOUL_VIEW,
            tooltip=MAP_TOOLTIP,
            map_provider="carto",
            map_style="light"
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

        st.caption(
            f"색상은 법정동별 평균 환산월세의 "
            f"5-95분위({vmin:.1f}-{vmax:.1f}만원)를 "
            f"기준으로 하며 거래 {MIN_MAP_COUNT}건 미만 "
            f"지역은 제외합니다."
        )

        if show_count_labels:
            st.caption(
                f"지도 숫자는 법정동별 거래건수이며 "
                f"{MIN_DONG_LABEL_COUNT}건 이상인 지역만 "
                f"표시합니다."
            )

        display_df = (
            dong_map[
                [
                    "EMD_NM",
                    "거래건수",
                    "평균",
                    "중앙",
                    "최저",
                    "최고"
                ]
            ]
            .sort_values(
                "거래건수",
                ascending=False
            )
            .rename(
                columns={
                    "EMD_NM": "법정동",
                    "평균": "평균 환산월세",
                    "중앙": "중앙 환산월세",
                    "최저": "최저 환산월세",
                    "최고": "최고 환산월세"
                }
            )
        )

        with st.expander(
            "📋 법정동별 상세 통계 보기"
        ):
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    # 20. 격자별
    elif spatial_unit == "격자별":
        grid = load_grid()

        grid_df = df[
            df["grid_id"].notna() &
            (df["grid_id"] != -1)
        ].copy()

        grid_df["grid_id"] = (
            grid_df["grid_id"].astype("Int64")
        )
        grid["grid_id"] = (
            grid["grid_id"].astype("Int64")
        )

        grid_stats = aggregate_rent(
            grid_df,
            ["grid_id"]
        )

        grid_map = grid.merge(
            grid_stats,
            on="grid_id",
            how="inner"
        )

        grid_map = grid_map[
            grid_map["거래건수"] >= MIN_MAP_COUNT
        ].copy()

        if grid_map.empty:
            st.warning(
                "거래 5건 이상인 격자가 없습니다."
            )
            st.stop()

        grid_map, vmin, vmax = add_map_color(
            grid_map
        )

        grid_map["tip_title"] = (
            "500m 격자 "
            + grid_map["grid_id"]
            .astype(str)
        )
        grid_map["tip_1"] = (
            "거래건수 "
            + grid_map["거래건수"]
            .astype(int).astype(str)
            + "건"
        )
        grid_map["tip_2"] = (
            "평균 환산월세 "
            + grid_map["평균"]
            .map(lambda x: f"{x:.1f}만원")
        )
        grid_map["tip_3"] = (
            "중앙 환산월세 "
            + grid_map["중앙"]
            .map(lambda x: f"{x:.1f}만원")
        )
        grid_map["tip_4"] = (
            "최저 "
            + grid_map["최저"]
            .map(lambda x: f"{x:.1f}만원")
        )
        grid_map["tip_5"] = (
            "최고 "
            + grid_map["최고"]
            .map(lambda x: f"{x:.1f}만원")
        )

        grid_layer = pdk.Layer(
            "GeoJsonLayer",
            data=grid_map,
            filled=True,
            stroked=True,
            get_fill_color="fill_color",
            get_line_color=[90, 90, 90, 80],
            line_width_min_pixels=0.3,
            pickable=True,
            auto_highlight=True
        )

        layers = [grid_layer]

        if policy_layer is not None:
            layers.append(
                policy_layer
            )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=SEOUL_VIEW,
            tooltip=MAP_TOOLTIP,
            map_provider="carto",
            map_style="light"
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

        st.caption(
            f"색상은 500m 격자별 평균 환산월세의 "
            f"5-95분위({vmin:.1f}-{vmax:.1f}만원)를 "
            f"기준으로 하며 거래 {MIN_MAP_COUNT}건 미만 "
            f"격자는 제외합니다."
        )

        display_df = (
            grid_map[
                [
                    "grid_id",
                    "거래건수",
                    "평균",
                    "중앙",
                    "최저",
                    "최고"
                ]
            ]
            .sort_values(
                "거래건수",
                ascending=False
            )
            .rename(
                columns={
                    "평균": "평균 환산월세",
                    "중앙": "중앙 환산월세",
                    "최저": "최저 환산월세",
                    "최고": "최고 환산월세"
                }
            )
        )

        with st.expander(
            "📋 격자별 상세 통계 보기"
        ):
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    # 21. 지하철역별
    elif spatial_unit == "지하철역별":
        rent_station = load_rent_station()
        subway = load_subway()
        gu = load_gu()

        station_match = rent_station.merge(
            df[
                [
                    "rent_id",
                    "환산월세(만원)"
                ]
            ],
            on="rent_id",
            how="inner"
        )

        if station_match.empty:
            st.warning(
                "선택한 조건에 해당하는 역세권 거래가 없습니다."
            )
            st.stop()

        station_stats = aggregate_rent(
            station_match,
            ["station_id"]
        )

        station_map = subway.merge(
            station_stats,
            on="station_id",
            how="inner"
        )

        if selected_subway_lines:
            station_map = station_map[
                station_map["hoseon"].apply(
                    lambda x: any(
                        selected_line in [
                            item.strip()
                            for item in str(x).split(",")
                        ]
                        for selected_line
                        in selected_subway_lines
                    )
                )
            ].copy()

        station_map = station_map[
            station_map["거래건수"] >= MIN_MAP_COUNT
        ].copy()

        if station_map.empty:
            st.warning(
                "선택한 조건에 해당하는 거래 5건 이상의 "
                "지하철역이 없습니다."
            )
            st.stop()

        station_map, vmin, vmax = add_map_color(
            station_map
        )

        station_map["radius"] = (
            100
            + np.sqrt(
                station_map["거래건수"]
            ) * 15
        ).clip(
            lower=100,
            upper=400
        )

        station_map["tip_title"] = (
            station_map["station_name"]
        )
        station_map["tip_1"] = (
            station_map["hoseon"].astype(str)
        )
        station_map["tip_2"] = (
            "거래건수 "
            + station_map["거래건수"]
            .astype(int).astype(str)
            + "건"
        )
        station_map["tip_3"] = (
            "평균 환산월세 "
            + station_map["평균"]
            .map(lambda x: f"{x:.1f}만원")
        )
        station_map["tip_4"] = (
            "중앙 환산월세 "
            + station_map["중앙"]
            .map(lambda x: f"{x:.1f}만원")
        )
        station_map["tip_5"] = (
            "500m 역세권"
        )

        station_data = pd.DataFrame(
            station_map.drop(
                columns="geometry",
                errors="ignore"
            )
        )

        gu_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gu,
            filled=False,
            stroked=True,
            get_line_color=[80, 80, 80, 80],
            line_width_min_pixels=0.7,
            pickable=False
        )

        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=station_data,
            get_position=[
                "longitude",
                "latitude"
            ],
            get_fill_color="fill_color",
            get_line_color=[70, 70, 70, 160],
            get_radius="radius",
            radius_min_pixels=4,
            radius_max_pixels=14,
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True
        )

        layers = [
            gu_layer,
            station_layer
        ]

        if policy_layer is not None:
            layers.append(
                policy_layer
            )

        if selected_subway_lines:
            station_text_layer = pdk.Layer(
                "TextLayer",
                data=station_data,
                get_position=[
                    "longitude",
                    "latitude"
                ],
                get_text="station_name",
                get_size=11,
                get_color=[70, 70, 70, 180],
                get_pixel_offset=[0, -12],
                get_text_anchor='"middle"',
                get_alignment_baseline='"bottom"',
                pickable=False
            )

            layers.append(
                station_text_layer
            )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=SEOUL_VIEW,
            tooltip=MAP_TOOLTIP,
            map_provider="carto",
            map_style="light"
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

        if selected_subway_lines:
            line_text = ", ".join(
                selected_subway_lines
            )

            st.caption(
                f"{line_text} 역의 500m 역세권 기준입니다. "
                f"색상은 평균 환산월세의 5-95분위"
                f"({vmin:.1f}-{vmax:.1f}만원), "
                f"원의 크기는 거래건수를 나타냅니다."
            )

        else:
            st.caption(
                f"서울 지하철역의 500m 역세권 기준입니다. "
                f"색상은 평균 환산월세의 5-95분위"
                f"({vmin:.1f}-{vmax:.1f}만원), "
                f"원의 크기는 거래건수를 나타냅니다."
            )

        display_df = (
            station_map[
                [
                    "station_name",
                    "hoseon",
                    "거래건수",
                    "평균",
                    "중앙",
                    "최저",
                    "최고"
                ]
            ]
            .sort_values(
                "거래건수",
                ascending=False
            )
            .rename(
                columns={
                    "station_name": "역명",
                    "hoseon": "노선",
                    "평균": "평균 환산월세",
                    "중앙": "중앙 환산월세",
                    "최저": "최저 환산월세",
                    "최고": "최고 환산월세"
                }
            )
        )

        with st.expander(
            "🚉 지하철역별 상세 통계 보기"
        ):
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    # 22. LH 청년매입임대 결과
    if show_policy:
        st.divider()
        st.subheader(
            "🏠 조건에 맞는 LH 청년매입임대"
        )

        st.caption(
            "정책주택은 전용면적·층수·주택유형 조건을 적용합니다. "
            "공고 자료에 건축년도 정보가 없어 건물 연식 조건은 "
            "정책주택 필터에 적용하지 않습니다."
        )

        if policy_units_filtered.empty:
            st.info(
                "현재 면적·층수·주택유형 조건에 맞는 "
                "청년매입임대 공급호실이 없습니다."
            )

        else:
            policy_metric1, policy_metric2, policy_metric3 = (
                st.columns(3)
            )

            policy_metric1.metric(
                "조건에 맞는 건물",
                f"{policy_map['building_id'].nunique():,}개"
            )

            policy_metric2.metric(
                "조건에 맞는 공급호실",
                f"{len(policy_units_filtered):,}호"
            )

            comparable_units = (
                policy_units_filtered[
                    "정책월세_만원"
                ].notna()
            )

            if comparable_units.any():
                policy_median = (
                    policy_units_filtered.loc[
                        comparable_units,
                        "정책월세_만원"
                    ].median()
                )

                policy_metric3.metric(
                    f"{policy_priority} 전환월세 중앙",
                    f"{policy_median:.1f}만원"
                )

            else:
                policy_metric3.metric(
                    f"{policy_priority} 전환월세",
                    "비교 불가"
                )

            st.caption(
                f"보라색 점은 조건에 맞는 LH 청년매입임대입니다. "
                f"월세 비교값은 {policy_priority}의 공고상 "
                f"'기본 임대조건-임대료→보증금 최대전환' 두 지점 "
                f"사이를 선형 보간하여 보증금 "
                f"{base_deposit:,}만원 기준으로 계산했습니다. "
                f"민간 거래에 사용하는 0.5% 환산식은 "
                f"정책주택에 적용하지 않았습니다."
            )

            if (
                policy_units_filtered[
                    "정책월세_만원"
                ].isna().any()
            ):
                st.caption(
                    "선택한 보증금이 해당 호실의 LH 공식 전환 가능 "
                    "범위를 벗어나는 경우 전환월세는 계산하지 않았습니다."
                )

            policy_display = (
                policy_map[
                    [
                        "주택명",
                        "지오코딩주소",
                        "조건공급호수",
                        "비교가능호수",
                        "전용면적_최소",
                        "전용면적_최대",
                        "정책월세_중앙",
                        "정책월세_최저",
                        "정책월세_최고"
                    ]
                ]
                .sort_values(
                    [
                        "정책월세_중앙",
                        "조건공급호수"
                    ],
                    ascending=[
                        True,
                        False
                    ],
                    na_position="last"
                )
                .rename(
                    columns={
                        "지오코딩주소": "주소",
                        "조건공급호수": "조건 공급호수",
                        "비교가능호수": "월세 비교가능호수",
                        "전용면적_최소": "전용면적 최소",
                        "전용면적_최대": "전용면적 최대",
                        "정책월세_중앙": "전환월세 중앙",
                        "정책월세_최저": "전환월세 최저",
                        "정책월세_최고": "전환월세 최고"
                    }
                )
            )

            with st.expander(
                "🏘️ 정책주택 건물별 목록 보기"
            ):
                st.dataframe(
                    policy_display,
                    use_container_width=True,
                    hide_index=True
                )

            if policy_priority == "청년 1순위":
                detail_prefix = "청년1순위"
            else:
                detail_prefix = "청년23순위"

            policy_unit_display = (
                policy_units_filtered[
                    [
                        "주택명",
                        "지오코딩주소",
                        "동호수",
                        "전용면적",
                        "층",
                        "주택유형",
                        f"{detail_prefix}_기본보증금_만원",
                        f"{detail_prefix}_기본월세_만원",
                        f"{detail_prefix}_최대전환보증금_만원",
                        f"{detail_prefix}_최대전환월세_만원",
                        "정책월세_만원"
                    ]
                ]
                .sort_values(
                    "정책월세_만원",
                    na_position="last"
                )
                .rename(
                    columns={
                        "지오코딩주소": "주소",
                        f"{detail_prefix}_기본보증금_만원":
                            "기본 보증금",
                        f"{detail_prefix}_기본월세_만원":
                            "기본 월세",
                        f"{detail_prefix}_최대전환보증금_만원":
                            "최대전환 보증금",
                        f"{detail_prefix}_최대전환월세_만원":
                            "최대전환 월세",
                        "정책월세_만원":
                            f"보증금 {base_deposit:,}만원 전환월세"
                    }
                )
            )

            with st.expander(
                "🚪 정책주택 호실별 임대조건 보기"
            ):
                st.dataframe(
                    policy_unit_display,
                    use_container_width=True,
                    hide_index=True
                )

    # 23. 시장 가격 구조
    st.divider()
    st.subheader("📈 서울 임대시장의 가격 구조")

    chart_col1, chart_col2 = st.columns(2)

    deposit_stats = make_deposit_stats(
        common_df,
        area_min,
        area_max
    )

    with chart_col1:
        st.markdown(
            "#### 보증금 수준별 월세"
        )

        st.caption(
            f"전용면적 {area_min}-{area_max}㎡의 "
            f"실제 월세 중앙값입니다."
        )

        deposit_chart = (
            alt.Chart(deposit_stats)
            .mark_line(
                point=True,
                strokeWidth=2
            )
            .encode(
                x=alt.X(
                    "보증금기준:N",
                    sort=list(
                        DEPOSIT_OPTIONS.keys()
                    ),
                    title="보증금 수준"
                ),
                y=alt.Y(
                    "중앙월세:Q",
                    title="월세 중앙값 (만원)"
                ),
                tooltip=[
                    alt.Tooltip(
                        "보증금기준:N",
                        title="보증금"
                    ),
                    alt.Tooltip(
                        "중앙월세:Q",
                        title="중앙 월세",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "평균월세:Q",
                        title="평균 월세",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "거래건수:Q",
                        title="거래건수",
                        format=","
                    )
                ]
            )
        )

        st.altair_chart(
            deposit_chart,
            use_container_width=True
        )

    area_stats = make_area_stats(
        common_df,
        dep_min,
        dep_max,
        base_deposit
    )

    with chart_col2:
        st.markdown(
            "#### 면적별 월세"
        )

        st.caption(
            f"보증금 {selected_deposit_label} 기준 "
            f"환산월세 중앙값입니다."
        )

        area_chart = (
            alt.Chart(area_stats)
            .mark_line(
                point=True,
                strokeWidth=2
            )
            .encode(
                x=alt.X(
                    "면적대:N",
                    title="전용면적 (㎡)"
                ),
                y=alt.Y(
                    "중앙월세:Q",
                    title="환산월세 중앙값 (만원)"
                ),
                tooltip=[
                    alt.Tooltip(
                        "면적대:N",
                        title="면적대"
                    ),
                    alt.Tooltip(
                        "중앙월세:Q",
                        title="중앙 환산월세",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "평균월세:Q",
                        title="평균 환산월세",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "거래건수:Q",
                        title="거래건수",
                        format=","
                    )
                ]
            )
        )

        st.altair_chart(
            area_chart,
            use_container_width=True
        )

else:
    st.info(
        "왼쪽 사이드바에서 조건을 직접 설정하거나 "
        "빠른 검색을 이용해 주세요."
    )
