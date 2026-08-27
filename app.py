import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np
import altair as alt

from pathlib import Path
from matplotlib import colormaps
from matplotlib.colors import Normalize


# ============================================================
# 0. BUILD
# ============================================================

APP_BUILD = "reset-20260827-02"


# ============================================================
# 1. 페이지 설정
# ============================================================

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

st.sidebar.caption(
    f"build: {APP_BUILD}"
)


# ============================================================
# 2. 데이터 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 3. 데이터 로드
# ============================================================

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
def load_subway_buffer():
    gdf = gpd.read_file(
        DATA_DIR / "subway_500m_buffer.geojson"
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


# ============================================================
# 4. 기본 설정
# ============================================================

MIN_MAP_COUNT = 5
MIN_DONG_LABEL_COUNT = 20


# ------------------------------------------------------------
# 지하철 노선 대표 색상
# ------------------------------------------------------------

LINE_COLOR_MAP = {
    "1호선": "#0052A4",
    "2호선": "#00A84D",
    "3호선": "#EF7C1C",
    "4호선": "#00A5DE",
    "5호선": "#996CAC",
    "6호선": "#CD7C2F",
    "7호선": "#747F00",
    "8호선": "#E6186C",
    "9호선": "#BDB092",

    "공항철도": "#0090D2",

    "경의중앙선": "#77C4A3",
    "경의·중앙선": "#77C4A3",

    "경춘선": "#0C8E72",

    "수인분당선": "#F5A200",
    "분당선": "#F5A200",

    "신분당선": "#D4003B",

    "우이신설선": "#B7C452",
    "신림선": "#6789CA",
    "서해선": "#8FC31F",

    "김포골드라인": "#AD8605",
    "에버라인": "#56AD2D",

    "인천1호선": "#7CA8D5",
    "인천2호선": "#ED8B00"
}


# ------------------------------------------------------------
# 지하철 전체 노선 목록
# ------------------------------------------------------------

subway_reference = load_subway()

all_subway_lines = sorted({
    line.strip()
    for text in subway_reference["hoseon"].dropna()
    for line in str(text).split(",")
})


# ------------------------------------------------------------
# 보증금 기준
# ------------------------------------------------------------

DEPOSIT_BASE_VALUES = [
    0,
    500,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000
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

    return (
        base_deposit,
        base_deposit + 1000
    )


# ============================================================
# 5. 기본 위젯값
# ============================================================

st.session_state.setdefault(
    "house_type_widget",
    "전체"
)

st.session_state.setdefault(
    "spatial_unit_widget",
    "법정동별"
)

st.session_state.setdefault(
    "subway_lines_widget",
    []
)

# 기본 연도
st.session_state.setdefault(
    "year_widget",
    2025
)

# 기본 보증금
st.session_state.setdefault(
    "deposit_widget",
    ["1,000만원"]
)

# 기본 면적 15-40㎡
st.session_state.setdefault(
    "area_widget",
    (15, 40)
)

# 기본 건물연식 전체
st.session_state.setdefault(
    "age_widget",
    (0, 100)
)

# 기본 층수 전체
st.session_state.setdefault(
    "floor_widget",
    "전체"
)

st.session_state.setdefault(
    "count_label_widget",
    False
)

st.session_state.setdefault(
    "show_policy_widget",
    True
)

st.session_state.setdefault(
    "policy_priority_widget",
    "청년 1순위"
)


# 예전 단일 선택 세션이 남아 있을 경우
# 다중 선택 형태로 자동 변환
if isinstance(
    st.session_state.get(
        "deposit_widget"
    ),
    str
):
    st.session_state[
        "deposit_widget"
    ] = [
        st.session_state[
            "deposit_widget"
        ]
    ]


# ============================================================
# 6. 프리셋
# ============================================================

def apply_preset(
    deposit,
    area,
    age,
    floor="지하·반지하 제외",
    house_type="전체"
):

    st.session_state[
        "house_type_widget"
    ] = house_type

    st.session_state[
        "spatial_unit_widget"
    ] = "법정동별"

    st.session_state[
        "subway_lines_widget"
    ] = []

    st.session_state[
        "year_widget"
    ] = 2025

    st.session_state[
        "deposit_widget"
    ] = (
        deposit
        if isinstance(
            deposit,
            list
        )
        else [deposit]
    )

    st.session_state[
        "area_widget"
    ] = area

    st.session_state[
        "age_widget"
    ] = age

    st.session_state[
        "floor_widget"
    ] = floor

    st.session_state[
        "show_policy_widget"
    ] = True

    st.session_state[
        "policy_priority_widget"
    ] = "청년 1순위"

    st.session_state[
        "preset_run"
    ] = True


def reset_filters():

    st.session_state[
        "house_type_widget"
    ] = "전체"

    st.session_state[
        "spatial_unit_widget"
    ] = "법정동별"

    st.session_state[
        "subway_lines_widget"
    ] = []

    st.session_state[
        "year_widget"
    ] = 2025

    st.session_state[
        "deposit_widget"
    ] = [
        "1,000만원"
    ]

    st.session_state[
        "area_widget"
    ] = (
        15,
        40
    )

    st.session_state[
        "age_widget"
    ] = (
        0,
        100
    )

    st.session_state[
        "floor_widget"
    ] = "전체"

    st.session_state[
        "count_label_widget"
    ] = False

    st.session_state[
        "show_policy_widget"
    ] = True

    st.session_state[
        "policy_priority_widget"
    ] = "청년 1순위"

    st.session_state[
        "preset_run"
    ] = False


# ============================================================
# 7. 검색 조건
# ============================================================

with st.sidebar.form(
    "search_form"
):

    st.header(
        "🔍 검색 조건 설정"
    )

    house_type_selection = st.radio(
        "주택 유형",
        [
            "전체",
            "연립다세대",
            "오피스텔"
        ],
        key="house_type_widget"
    )

    spatial_unit = st.radio(
        "시각화 단위",
        [
            "법정동별",
            "격자별",
            "지하철역별"
        ],
        key="spatial_unit_widget"
    )

    selected_subway_lines = (
        st.multiselect(
            "지하철 노선",
            options=all_subway_lines,
            key="subway_lines_widget",
            help=(
                "지하철역별 지도에 적용됩니다. "
                "선택하지 않으면 전체 역을 표시합니다. "
                "아래 노선별 비교에는 이 필터가 적용되지 않습니다."
            )
        )
    )

    selected_year = st.selectbox(
        "연도",
        [
            2025,
            2024,
            2023
        ],
        key="year_widget"
    )

    selected_deposit_labels = (
        st.multiselect(
            "기준 보증금",
            options=list(
                DEPOSIT_OPTIONS.keys()
            ),
            key="deposit_widget",
            help=(
                "여러 보증금 기준을 함께 선택할 수 있습니다. "
                "예를 들어 500·1,000·2,000만원을 선택하면 "
                "1-999만원은 500만원, "
                "1,000-1,999만원은 1,000만원, "
                "2,000-2,999만원은 2,000만원 기준으로 "
                "각각 환산합니다."
            )
        )
    )

    area_min, area_max = st.slider(
        "임대면적 (㎡)",
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
            "지하·반지하 제외"
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

    st.markdown(
        "##### 🏠 정책주택"
    )

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
        [
            "청년 1순위",
            "청년 2·3순위"
        ],
        key="policy_priority_widget",
        disabled=not show_policy
    )

    submit_button = (
        st.form_submit_button(
            "우리집 찾기",
            type="primary",
            use_container_width=True
        )
    )


# ============================================================
# 8. 빠른 검색
# ============================================================

st.sidebar.markdown(
    "### 🏠 빠른 검색"
)

st.sidebar.caption(
    "자주 찾는 조건을 한 번에 적용합니다."
)


preset_col1, preset_col2 = (
    st.sidebar.columns(2)
)


preset_col1.button(
    "1000 · 5-7평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "1,000만원",
        (16, 23),
        (0, 100)
    )
)


preset_col2.button(
    "1000 · 8-12평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "1,000만원",
        (26, 40),
        (0, 100)
    )
)


preset_col3, preset_col4 = (
    st.sidebar.columns(2)
)


preset_col3.button(
    "1000 · 신축",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "1,000만원",
        (26, 40),
        (0, 10)
    )
)


preset_col4.button(
    "2000 · 8-12평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "2,000만원",
        (26, 40),
        (0, 100)
    )
)


preset_col5, preset_col6 = (
    st.sidebar.columns(2)
)


preset_col5.button(
    "3000 · 8-12평",
    use_container_width=True,
    on_click=apply_preset,
    args=(
        "3,000만원",
        (26, 40),
        (0, 100)
    )
)


preset_col6.button(
    "조건 초기화",
    use_container_width=True,
    on_click=reset_filters
)


run_search = (
    submit_button
    or st.session_state.pop(
        "preset_run",
        False
    )
)


# ============================================================
# 9. 민간 임대 공통 필터
# ============================================================

def filter_common_data(
    df,
    house_type,
    contract_year,
    age_min,
    age_max,
    floor
):

    df = df[
        df["전월세구분"]
        == "월세"
    ].copy()

    if house_type != "전체":

        df = df[
            df["건물용도"]
            == house_type
        ].copy()

    df["건물연식"] = (
        contract_year
        - df["건축년도"]
    )

    df = df[
        (
            df["건물연식"]
            >= age_min
        )
        &
        (
            df["건물연식"]
            <= age_max
        )
    ].copy()

    if floor == "지하·반지하 제외":

        df = df[
            df["층"] > 0
        ].copy()

    return df


# ============================================================
# 10. 다중 보증금 환산
# ============================================================

def filter_rent_data(
    df,
    base_deposits,
    area_min,
    area_max
):

    parts = []

    for base_deposit in base_deposits:

        dep_min, dep_max = (
            get_deposit_range(
                base_deposit
            )
        )

        part = df[
            (
                df["보증금(만원)"]
                >= dep_min
            )
            &
            (
                df["보증금(만원)"]
                < dep_max
            )
            &
            (
                df["임대면적"]
                >= area_min
            )
            &
            (
                df["임대면적"]
                <= area_max
            )
        ].copy()

        if part.empty:
            continue

        part[
            "기준보증금(만원)"
        ] = base_deposit

        part[
            "환산월세(만원)"
        ] = (
            part[
                "임대료(만원)"
            ]
            +
            (
                part[
                    "보증금(만원)"
                ]
                - base_deposit
            )
            * 0.005
        )

        parts.append(
            part
        )

    if not parts:

        empty = (
            df.iloc[
                0:0
            ].copy()
        )

        empty[
            "기준보증금(만원)"
        ] = pd.Series(
            dtype="float64"
        )

        empty[
            "환산월세(만원)"
        ] = pd.Series(
            dtype="float64"
        )

        return empty

    return pd.concat(
        parts,
        ignore_index=True
    )


def make_selected_deposit_caption(
    base_deposits
):

    pieces = []

    for base_deposit in (
        sorted(
            base_deposits
        )
    ):

        dep_min, dep_max = (
            get_deposit_range(
                base_deposit
            )
        )

        if base_deposit == 0:

            pieces.append(
                "보증금 0만원 → 무보증 기준"
            )

        else:

            pieces.append(
                f"{dep_min:,}-{dep_max - 1:,}만원 "
                f"→ {base_deposit:,}만원 기준"
            )

    return " / ".join(
        pieces
    )


# ============================================================
# 11. 공간 단위별 통계
# ============================================================

def aggregate_rent(
    df,
    group_cols
):

    stats = (
        df.groupby(
            group_cols
        )[
            "환산월세(만원)"
        ]
        .agg(
            거래건수="count",
            평균="mean",
            중앙="median",
            최저="min",
            최고="max"
        )
        .reset_index()
    )

    for col in [
        "평균",
        "중앙",
        "최저",
        "최고"
    ]:

        stats[col] = (
            stats[col]
            .round(1)
        )

    return stats


# ============================================================
# 12. 지도 색상
# ============================================================

def add_map_color(
    data,
    value_col="평균"
):

    valid_values = (
        data[value_col]
        .dropna()
    )

    vmin = (
        valid_values
        .quantile(0.05)
    )

    vmax = (
        valid_values
        .quantile(0.95)
    )

    if vmin == vmax:
        vmax = vmin + 1

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=True
    )

    cmap = colormaps[
        "RdYlBu_r"
    ]

    def make_color(value):

        if pd.isna(value):

            return [
                220,
                220,
                220,
                0
            ]

        rgba = cmap(
            norm(value)
        )

        return [
            int(
                rgba[0] * 255
            ),
            int(
                rgba[1] * 255
            ),
            int(
                rgba[2] * 255
            ),
            190
        ]

    data = data.copy()

    data[
        "fill_color"
    ] = (
        data[value_col]
        .apply(
            make_color
        )
    )

    return (
        data,
        vmin,
        vmax
    )


# ============================================================
# 13. 법정동 거래건수 라벨
# ============================================================

def make_dong_label_data(
    dong_map,
    min_count=MIN_DONG_LABEL_COUNT
):

    label_gdf = (
        dong_map[
            dong_map[
                "거래건수"
            ] >= min_count
        ][
            [
                "EMD_NM",
                "거래건수",
                "geometry"
            ]
        ]
        .copy()
    )

    if label_gdf.empty:
        return pd.DataFrame()

    label_gdf = (
        label_gdf
        .to_crs(
            epsg=5179
        )
    )

    label_gdf[
        "geometry"
    ] = (
        label_gdf
        .geometry
        .representative_point()
    )

    label_gdf = (
        label_gdf
        .to_crs(
            epsg=4326
        )
    )

    label_gdf[
        "longitude"
    ] = (
        label_gdf.geometry.x
    )

    label_gdf[
        "latitude"
    ] = (
        label_gdf.geometry.y
    )

    label_gdf[
        "label"
    ] = (
        label_gdf[
            "거래건수"
        ]
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


# ============================================================
# 14. 보증금 수준별 월세 통계
# ============================================================

def make_deposit_stats(
    df,
    area_min,
    area_max
):

    chart_df = df[
        (
            df["임대면적"]
            >= area_min
        )
        &
        (
            df["임대면적"]
            <= area_max
        )
    ].copy()

    chart_df[
        "보증금기준"
    ] = pd.NA

    chart_df.loc[
        chart_df[
            "보증금(만원)"
        ] == 0,
        "보증금기준"
    ] = "무보증"

    chart_df.loc[
        (
            chart_df[
                "보증금(만원)"
            ] >= 1
        )
        &
        (
            chart_df[
                "보증금(만원)"
            ] < 1000
        ),
        "보증금기준"
    ] = "500만원"

    for base in range(
        1000,
        11000,
        1000
    ):

        chart_df.loc[
            (
                chart_df[
                    "보증금(만원)"
                ] >= base
            )
            &
            (
                chart_df[
                    "보증금(만원)"
                ] < base + 1000
            ),
            "보증금기준"
        ] = (
            f"{base:,}만원"
        )

    chart_df[
        "보증금기준"
    ] = pd.Categorical(
        chart_df[
            "보증금기준"
        ],
        categories=list(
            DEPOSIT_OPTIONS.keys()
        ),
        ordered=True
    )

    stats = (
        chart_df
        .dropna(
            subset=[
                "보증금기준"
            ]
        )
        .groupby(
            "보증금기준",
            observed=True
        )
        .agg(
            거래건수=(
                "임대료(만원)",
                "count"
            ),
            평균월세=(
                "임대료(만원)",
                "mean"
            ),
            중앙월세=(
                "임대료(만원)",
                "median"
            )
        )
        .reset_index()
    )

    stats[
        "평균월세"
    ] = (
        stats[
            "평균월세"
        ].round(1)
    )

    stats[
        "중앙월세"
    ] = (
        stats[
            "중앙월세"
        ].round(1)
    )

    return stats


# ============================================================
# 15. 면적별 환산월세 통계
# ============================================================

AREA_BINS = list(
    range(
        5,
        95,
        5
    )
)

AREA_BIN_ORDER = [
    f"{start}-{start + 5}"
    for start
    in AREA_BINS[:-1]
]


def make_area_stats(
    df,
    selected_bases
):

    chart_df = (
        filter_rent_data(
            df,
            selected_bases,
            5,
            85
        )
    )

    if chart_df.empty:
        return pd.DataFrame()

    chart_df[
        "면적대"
    ] = pd.cut(
        chart_df[
            "임대면적"
        ],
        bins=AREA_BINS,
        labels=AREA_BIN_ORDER,
        right=False
    )

    stats = (
        chart_df
        .dropna(
            subset=[
                "면적대"
            ]
        )
        .groupby(
            "면적대",
            observed=True
        )
        .agg(
            거래건수=(
                "환산월세(만원)",
                "count"
            ),
            평균월세=(
                "환산월세(만원)",
                "mean"
            ),
            중앙월세=(
                "환산월세(만원)",
                "median"
            )
        )
        .reset_index()
    )

    stats[
        "면적대"
    ] = (
        stats[
            "면적대"
        ].astype(str)
    )

    stats[
        "평균월세"
    ] = (
        stats[
            "평균월세"
        ].round(1)
    )

    stats[
        "중앙월세"
    ] = (
        stats[
            "중앙월세"
        ].round(1)
    )

    return stats


# ============================================================
# 16. 지하철 노선별 통계
# ============================================================

def make_subway_line_stats(
    station_match,
    subway
):

    station_lines = (
        subway[
            [
                "station_id",
                "hoseon"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    station_lines[
        "station_id"
    ] = (
        station_lines[
            "station_id"
        ]
        .astype(str)
    )

    station_lines[
        "노선"
    ] = (
        station_lines[
            "hoseon"
        ]
        .astype(str)
        .str.split(",")
    )

    station_lines = (
        station_lines
        .explode(
            "노선"
        )
    )

    station_lines[
        "노선"
    ] = (
        station_lines[
            "노선"
        ]
        .astype(str)
        .str.strip()
    )

    station_match = (
        station_match.copy()
    )

    station_match[
        "station_id"
    ] = (
        station_match[
            "station_id"
        ]
        .astype(str)
    )

    line_match = (
        station_match[
            [
                "rent_id",
                "station_id",
                "환산월세(만원)"
            ]
        ]
        .merge(
            station_lines[
                [
                    "station_id",
                    "노선"
                ]
            ],
            on="station_id",
            how="left"
        )
    )

    # 동일 거래가 동일 노선의
    # 여러 역 500m에 포함되는 경우 중복 제거
    line_match = (
        line_match
        .dropna(
            subset=[
                "노선"
            ]
        )
        .drop_duplicates(
            subset=[
                "rent_id",
                "노선"
            ]
        )
    )

    stats = (
        line_match
        .groupby(
            "노선"
        )
        .agg(
            거래건수=(
                "rent_id",
                "nunique"
            ),
            평균=(
                "환산월세(만원)",
                "mean"
            ),
            중앙=(
                "환산월세(만원)",
                "median"
            )
        )
        .reset_index()
    )

    stats = stats[
        stats[
            "거래건수"
        ] >= MIN_MAP_COUNT
    ].copy()

    stats[
        "평균"
    ] = (
        stats[
            "평균"
        ].round(1)
    )

    stats[
        "중앙"
    ] = (
        stats[
            "중앙"
        ].round(1)
    )

    return (
        stats
        .sort_values(
            "평균",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 17. LH 정책주택
# ============================================================

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

    units = (
        policy_units.copy()
    )

    buildings = (
        policy_buildings.copy()
    )

    # ID 자료형 통일
    units[
        "building_id"
    ] = (
        units[
            "building_id"
        ]
        .astype(str)
        .str.strip()
    )

    buildings[
        "building_id"
    ] = (
        buildings[
            "building_id"
        ]
        .astype(str)
        .str.strip()
    )

    buildings[
        "longitude"
    ] = pd.to_numeric(
        buildings[
            "longitude"
        ],
        errors="coerce"
    )

    buildings[
        "latitude"
    ] = pd.to_numeric(
        buildings[
            "latitude"
        ],
        errors="coerce"
    )

    # 정책주택은 전용면적 기준
    units = units[
        (
            units[
                "전용면적"
            ] >= area_min
        )
        &
        (
            units[
                "전용면적"
            ] <= area_max
        )
    ].copy()

    if house_type == "오피스텔":

        units = units[
            units[
                "주택유형"
            ] == "오피스텔"
        ].copy()

    elif house_type == "연립다세대":

        units = units[
            units[
                "주택유형"
            ]
            .astype(str)
            .str.contains(
                "다세대|연립",
                regex=True,
                na=False
            )
        ].copy()

    if floor == "지하·반지하 제외":

        units = units[
            units[
                "층"
            ] > 0
        ].copy()

    if units.empty:

        return (
            units,
            pd.DataFrame()
        )

    if priority == "청년 1순위":
        prefix = "청년1순위"

    else:
        prefix = "청년23순위"

    basic_dep_col = (
        f"{prefix}_기본보증금_만원"
    )

    basic_rent_col = (
        f"{prefix}_기본월세_만원"
    )

    max_dep_col = (
        f"{prefix}_최대전환보증금_만원"
    )

    max_rent_col = (
        f"{prefix}_최대전환월세_만원"
    )

    for col in [
        basic_dep_col,
        basic_rent_col,
        max_dep_col,
        max_rent_col
    ]:

        units[col] = pd.to_numeric(
            units[col],
            errors="coerce"
        )

    units[
        "정책월세_만원"
    ] = np.nan

    units[
        "비교가능"
    ] = False

    # 정책주택 공식 전환월세는
    # 보증금 기준을 하나만 선택한 경우 계산
    if base_deposit is not None:

        dep_gap = (
            units[
                max_dep_col
            ]
            -
            units[
                basic_dep_col
            ]
        )

        comparable = (
            units[
                basic_dep_col
            ].notna()
            &
            units[
                basic_rent_col
            ].notna()
            &
            units[
                max_dep_col
            ].notna()
            &
            units[
                max_rent_col
            ].notna()
            &
            (
                dep_gap > 0
            )
            &
            (
                base_deposit
                >= units[
                    basic_dep_col
                ]
            )
            &
            (
                base_deposit
                <= units[
                    max_dep_col
                ]
            )
        )

        units.loc[
            comparable,
            "정책월세_만원"
        ] = (
            units.loc[
                comparable,
                basic_rent_col
            ]
            +
            (
                base_deposit
                -
                units.loc[
                    comparable,
                    basic_dep_col
                ]
            )
            *
            (
                units.loc[
                    comparable,
                    max_rent_col
                ]
                -
                units.loc[
                    comparable,
                    basic_rent_col
                ]
            )
            /
            dep_gap.loc[
                comparable
            ]
        )

        units[
            "비교가능"
        ] = comparable

    units[
        "정책월세_만원"
    ] = (
        units[
            "정책월세_만원"
        ].round(1)
    )

    policy_stats = (
        units
        .groupby(
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

        policy_stats[
            col
        ] = (
            policy_stats[
                col
            ].round(1)
        )

    policy_map = (
        buildings[
            [
                "building_id",
                "주택명",
                "지오코딩주소",
                "longitude",
                "latitude"
            ]
        ]
        .merge(
            policy_stats,
            on="building_id",
            how="inner"
        )
    )

    policy_map = policy_map[
        policy_map[
            "longitude"
        ].notna()
        &
        policy_map[
            "latitude"
        ].notna()
    ].copy()

    policy_map[
        "tip_title"
    ] = (
        policy_map[
            "주택명"
        ]
    )

    policy_map[
        "tip_1"
    ] = (
        "LH 청년매입임대 · "
        +
        policy_map[
            "조건공급호수"
        ]
        .astype(int)
        .astype(str)
        +
        "호"
    )

    policy_map[
        "tip_2"
    ] = (
        "전용 "
        +
        policy_map[
            "전용면적_최소"
        ].map(
            lambda x:
                f"{x:.1f}"
        )
        +
        "-"
        +
        policy_map[
            "전용면적_최대"
        ].map(
            lambda x:
                f"{x:.1f}"
        )
        +
        "㎡"
    )

    if base_deposit is None:

        policy_map[
            "tip_3"
        ] = (
            priority
            +
            " · 다중 보증금 선택"
        )

        policy_map[
            "tip_4"
        ] = (
            "전환월세는 기준 보증금 "
            "1개 선택 시 계산"
        )

    else:

        policy_map[
            "tip_3"
        ] = (
            priority
            +
            f" · 보증금 "
            f"{base_deposit:,}만원 기준"
        )

        policy_map[
            "tip_4"
        ] = (
            policy_map[
                "정책월세_중앙"
            ]
            .apply(
                lambda x:
                    (
                        f"LH 전환월세 중앙 "
                        f"{x:.1f}만원"
                    )
                    if pd.notna(x)
                    else (
                        "선택 보증금은 "
                        "공식 전환범위 밖"
                    )
            )
        )

    policy_map[
        "tip_5"
    ] = (
        policy_map[
            "지오코딩주소"
        ]
    )

    return (
        units,
        policy_map
    )


def make_square(row):

    lon = row["longitude"]
    lat = row["latitude"]

    # 중심에서 각 방향으로 12m
    # → 전체 약 24m × 24m 사각형
    half_size_m = 12

    # 위도 1도당 약 111.32km
    lat_delta = (
        half_size_m
        / 111320
    )

    # 경도는 위도에 따라 실제 거리가 달라짐
    lon_delta = (
        half_size_m
        / (
            111320
            * np.cos(
                np.radians(lat)
            )
        )
    )

    return [
        [
            lon - lon_delta,
            lat - lat_delta
        ],
        [
            lon + lon_delta,
            lat - lat_delta
        ],
        [
            lon + lon_delta,
            lat + lat_delta
        ],
        [
            lon - lon_delta,
            lat + lat_delta
        ]
    ]

# ============================================================
# 18. 지도 툴팁 및 기본 View
# ============================================================

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


SEOUL_VIEW = pdk.ViewState(
    latitude=37.5665,
    longitude=126.9780,
    zoom=10.3,
    pitch=0
)


# ============================================================
# 19. 검색 실행
# ============================================================

if run_search:

    if not selected_deposit_labels:

        st.warning(
            "기준 보증금을 하나 이상 선택해 주세요."
        )

        st.stop()

    selected_base_deposits = sorted([
        DEPOSIT_OPTIONS[
            label
        ]
        for label
        in selected_deposit_labels
    ])

    selected_deposit_text = (
        ", ".join(
            deposit_label(
                value
            )
            for value
            in selected_base_deposits
        )
    )

    policy_base_deposit = (
        selected_base_deposits[0]
        if len(
            selected_base_deposits
        ) == 1
        else None
    )

    with st.spinner(
        f"{selected_year}년 거래 데이터를 불러오는 중..."
    ):

        df_raw = (
            load_rent(
                selected_year
            )
        )

    common_df = (
        filter_common_data(
            df_raw,
            house_type_selection,
            selected_year,
            age_min,
            age_max,
            selected_floor
        )
    )

    df = (
        filter_rent_data(
            common_df,
            selected_base_deposits,
            area_min,
            area_max
        )
    )

    if df.empty:

        st.warning(
            "선택한 조건에 해당하는 거래가 없습니다."
        )

        st.stop()


    # ========================================================
    # 정책주택
    # ========================================================

    policy_units_filtered = (
        pd.DataFrame()
    )

    policy_map = (
        pd.DataFrame()
    )

    policy_layer = None

    if show_policy:

        policy_units = (
            load_policy_units()
        )

        policy_buildings = (
            load_policy_buildings()
        )

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
            policy_base_deposit
        )

        policy_layer = (
            make_policy_layer(
                policy_map
            )
        )


    # ========================================================
    # 결과 요약
    # ========================================================

    st.subheader(
        f"📊 {selected_year}년 "
        f"{house_type_selection} "
        f"{spatial_unit}"
    )

    st.markdown(
        f"**기준 보증금 {selected_deposit_text} · "
        f"임대면적 {area_min}-{area_max}㎡ · "
        f"건물 연식 {age_min}-{age_max}년**"
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    metric1.metric(
        "거래건수",
        f"{len(df):,}건"
    )

    metric2.metric(
        "평균 환산월세",
        (
            f"{df['환산월세(만원)'].mean():.1f}"
            "만원"
        )
    )

    metric3.metric(
        "중앙 환산월세",
        (
            f"{df['환산월세(만원)'].median():.1f}"
            "만원"
        )
    )

    st.caption(
        make_selected_deposit_caption(
            selected_base_deposits
        )
    )


    # ========================================================
    # 20. 법정동별
    # ========================================================

    if spatial_unit == "법정동별":

        dong = load_dong()

        df[
            "자치구코드"
        ] = (
            df[
                "자치구코드"
            ].astype(
                "Int64"
            )
        )

        df[
            "법정동코드"
        ] = (
            df[
                "법정동코드"
            ].astype(
                "Int64"
            )
        )

        dong[
            "자치구코드"
        ] = (
            dong[
                "자치구코드"
            ].astype(
                "Int64"
            )
        )

        dong[
            "법정동코드"
        ] = (
            dong[
                "법정동코드"
            ].astype(
                "Int64"
            )
        )

        dong_stats = (
            aggregate_rent(
                df,
                [
                    "자치구코드",
                    "법정동코드"
                ]
            )
        )

        dong_map = (
            dong.merge(
                dong_stats,
                on=[
                    "자치구코드",
                    "법정동코드"
                ],
                how="inner"
            )
        )

        dong_map = dong_map[
            dong_map[
                "거래건수"
            ] >= MIN_MAP_COUNT
        ].copy()

        if dong_map.empty:

            st.warning(
                "거래 5건 이상인 법정동이 없습니다."
            )

            st.stop()

        (
            dong_map,
            vmin,
            vmax
        ) = add_map_color(
            dong_map
        )

        dong_map[
            "tip_title"
        ] = (
            dong_map[
                "EMD_NM"
            ]
        )

        dong_map[
            "tip_1"
        ] = (
            "거래건수 "
            +
            dong_map[
                "거래건수"
            ]
            .astype(int)
            .astype(str)
            +
            "건"
        )

        dong_map[
            "tip_2"
        ] = (
            "평균 환산월세 "
            +
            dong_map[
                "평균"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        dong_map[
            "tip_3"
        ] = (
            "중앙 환산월세 "
            +
            dong_map[
                "중앙"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        dong_map[
            "tip_4"
        ] = (
            "최저 "
            +
            dong_map[
                "최저"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        dong_map[
            "tip_5"
        ] = (
            "최고 "
            +
            dong_map[
                "최고"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        dong_layer = pdk.Layer(
            "GeoJsonLayer",
            data=dong_map,
            filled=True,
            stroked=True,
            get_fill_color="fill_color",
            get_line_color=[
                90,
                90,
                90,
                100
            ],
            line_width_min_pixels=0.5,
            pickable=True,
            auto_highlight=True
        )

        layers = [
            dong_layer
        ]

        if policy_layer is not None:

            layers.append(
                policy_layer
            )

        if show_count_labels:

            dong_label_data = (
                make_dong_label_data(
                    dong_map
                )
            )

            if not (
                dong_label_data.empty
            ):

                count_text_layer = (
                    pdk.Layer(
                        "TextLayer",
                        data=dong_label_data,
                        get_position=[
                            "longitude",
                            "latitude"
                        ],
                        get_text="label",
                        get_size=11,
                        get_color=[
                            40,
                            40,
                            40,
                            220
                        ],
                        get_text_anchor='"middle"',
                        get_alignment_baseline='"center"',
                        pickable=False
                    )
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
                    "EMD_NM":
                        "법정동",
                    "평균":
                        "평균 환산월세",
                    "중앙":
                        "중앙 환산월세",
                    "최저":
                        "최저 환산월세",
                    "최고":
                        "최고 환산월세"
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


    # ========================================================
    # 21. 격자별
    # ========================================================

    elif spatial_unit == "격자별":

        grid = load_grid()

        grid_df = df[
            df[
                "grid_id"
            ].notna()
            &
            (
                df[
                    "grid_id"
                ] != -1
            )
        ].copy()

        grid_df[
            "grid_id"
        ] = (
            grid_df[
                "grid_id"
            ].astype(
                "Int64"
            )
        )

        grid[
            "grid_id"
        ] = (
            grid[
                "grid_id"
            ].astype(
                "Int64"
            )
        )

        grid_stats = (
            aggregate_rent(
                grid_df,
                [
                    "grid_id"
                ]
            )
        )

        grid_map = (
            grid.merge(
                grid_stats,
                on="grid_id",
                how="inner"
            )
        )

        grid_map = grid_map[
            grid_map[
                "거래건수"
            ] >= MIN_MAP_COUNT
        ].copy()

        if grid_map.empty:

            st.warning(
                "거래 5건 이상인 격자가 없습니다."
            )

            st.stop()

        (
            grid_map,
            vmin,
            vmax
        ) = add_map_color(
            grid_map
        )

        grid_map[
            "tip_title"
        ] = (
            "500m 격자 "
            +
            grid_map[
                "grid_id"
            ].astype(str)
        )

        grid_map[
            "tip_1"
        ] = (
            "거래건수 "
            +
            grid_map[
                "거래건수"
            ]
            .astype(int)
            .astype(str)
            +
            "건"
        )

        grid_map[
            "tip_2"
        ] = (
            "평균 환산월세 "
            +
            grid_map[
                "평균"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        grid_map[
            "tip_3"
        ] = (
            "중앙 환산월세 "
            +
            grid_map[
                "중앙"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        grid_map[
            "tip_4"
        ] = (
            "최저 "
            +
            grid_map[
                "최저"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        grid_map[
            "tip_5"
        ] = (
            "최고 "
            +
            grid_map[
                "최고"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        grid_layer = pdk.Layer(
            "GeoJsonLayer",
            data=grid_map,
            filled=True,
            stroked=True,
            get_fill_color="fill_color",
            get_line_color=[
                90,
                90,
                90,
                80
            ],
            line_width_min_pixels=0.3,
            pickable=True,
            auto_highlight=True
        )

        layers = [
            grid_layer
        ]

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
                    "평균":
                        "평균 환산월세",
                    "중앙":
                        "중앙 환산월세",
                    "최저":
                        "최저 환산월세",
                    "최고":
                        "최고 환산월세"
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


    # ========================================================
    # 22. 지하철역별
    # ========================================================

    elif spatial_unit == "지하철역별":

        rent_station = (
            load_rent_station()
        )

        subway = (
            load_subway()
        )

        subway_buffer = (
            load_subway_buffer()
        )

        gu = load_gu()


        # ----------------------------------------------------
        # station_id 자료형 통일
        # ----------------------------------------------------

        rent_station[
            "station_id"
        ] = (
            rent_station[
                "station_id"
            ]
            .astype(str)
            .str.strip()
        )

        subway[
            "station_id"
        ] = (
            subway[
                "station_id"
            ]
            .astype(str)
            .str.strip()
        )

        subway_buffer[
            "station_id"
        ] = (
            subway_buffer[
                "station_id"
            ]
            .astype(str)
            .str.strip()
        )


        # ----------------------------------------------------
        # 현재 조건 거래와 역세권 대응
        # ----------------------------------------------------

        station_match = (
            rent_station.merge(
                df[
                    [
                        "rent_id",
                        "환산월세(만원)"
                    ]
                ],
                on="rent_id",
                how="inner"
            )
        )

        if station_match.empty:

            st.warning(
                "선택한 조건에 해당하는 "
                "역세권 거래가 없습니다."
            )

            st.stop()


        # ----------------------------------------------------
        # 노선별 통계
        # 지도 노선 선택과 무관하게 전체 노선 계산
        # ----------------------------------------------------

        line_stats = (
            make_subway_line_stats(
                station_match,
                subway
            )
        )


        # ----------------------------------------------------
        # 역별 통계
        # ----------------------------------------------------

        station_stats = (
            aggregate_rent(
                station_match,
                [
                    "station_id"
                ]
            )
        )

        station_map = (
            subway.merge(
                station_stats,
                on="station_id",
                how="inner"
            )
        )


        # ----------------------------------------------------
        # 지도에서만 선택 노선 적용
        # ----------------------------------------------------

        if selected_subway_lines:

            station_map = (
                station_map[
                    station_map[
                        "hoseon"
                    ]
                    .apply(
                        lambda x:
                            any(
                                selected_line
                                in [
                                    item.strip()
                                    for item
                                    in str(x).split(",")
                                ]
                                for selected_line
                                in selected_subway_lines
                            )
                    )
                ]
                .copy()
            )


        station_map = station_map[
            station_map[
                "거래건수"
            ] >= MIN_MAP_COUNT
        ].copy()


        if station_map.empty:

            st.warning(
                "선택한 조건에 해당하는 "
                "거래 5건 이상의 지하철역이 없습니다."
            )

            st.stop()


        # ----------------------------------------------------
        # 현재 지도에 표시되는 역의 500m 버퍼만 추출
        # ----------------------------------------------------

        visible_station_ids = set(
            station_map[
                "station_id"
            ]
            .dropna()
            .astype(str)
        )

        candidate_buffer = (
            subway_buffer[
                subway_buffer[
                    "station_id"
                ].isin(
                    visible_station_ids
                )
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Polygon을 LineString 경계로 변환
        #
        # 중요:
        # Polygon 자체를 PyDeck에 넘기지 않음.
        # 따라서 내부가 파랗게 채워지는 문제 방지.
        # ----------------------------------------------------

        buffer_map = None

        if not candidate_buffer.empty:

            candidate_buffer = (
                candidate_buffer[
                    ~candidate_buffer.geometry.is_empty
                    &
                    candidate_buffer.geometry.notna()
                ]
                .copy()
            )

            if not candidate_buffer.empty:

                projected = (
                    candidate_buffer
                    .to_crs(
                        epsg=5186
                    )
                    .copy()
                )

                polygon_mask = (
                    projected
                    .geometry
                    .geom_type
                    .isin(
                        [
                            "Polygon",
                            "MultiPolygon"
                        ]
                    )
                )

                polygon_part = (
                    projected[
                        polygon_mask
                    ].copy()
                )

                line_part = (
                    projected[
                        ~polygon_mask
                    ].copy()
                )


                # Polygon은 면적 검증 후 경계선만 사용
                if not polygon_part.empty:

                    polygon_part[
                        "_area_m2"
                    ] = (
                        polygon_part
                        .geometry
                        .area
                    )

                    polygon_part = (
                        polygon_part[
                            polygon_part[
                                "_area_m2"
                            ].between(
                                10000,
                                5000000
                            )
                        ]
                        .copy()
                    )

                    if not polygon_part.empty:

                        polygon_part[
                            "geometry"
                        ] = (
                            polygon_part
                            .geometry
                            .boundary
                        )


                # 이미 LineString인 파일에도 대응
                buffer_parts = []

                if not polygon_part.empty:

                    buffer_parts.append(
                        polygon_part
                    )

                if not line_part.empty:

                    buffer_parts.append(
                        line_part
                    )


                if buffer_parts:

                    buffer_map = (
                        pd.concat(
                            buffer_parts,
                            ignore_index=True
                        )
                    )

                    buffer_map = (
                        gpd.GeoDataFrame(
                            buffer_map,
                            geometry="geometry",
                            crs=projected.crs
                        )
                        .to_crs(
                            epsg=4326
                        )
                    )


        # ----------------------------------------------------
        # 역 마커 색상
        # ----------------------------------------------------

        (
            station_map,
            vmin,
            vmax
        ) = add_map_color(
            station_map
        )

        station_map[
            "radius"
        ] = (
            100
            +
            np.sqrt(
                station_map[
                    "거래건수"
                ]
            )
            * 15
        ).clip(
            lower=100,
            upper=400
        )

        station_map[
            "tip_title"
        ] = (
            station_map[
                "station_name"
            ]
        )

        station_map[
            "tip_1"
        ] = (
            station_map[
                "hoseon"
            ].astype(str)
        )

        station_map[
            "tip_2"
        ] = (
            "거래건수 "
            +
            station_map[
                "거래건수"
            ]
            .astype(int)
            .astype(str)
            +
            "건"
        )

        station_map[
            "tip_3"
        ] = (
            "평균 환산월세 "
            +
            station_map[
                "평균"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        station_map[
            "tip_4"
        ] = (
            "중앙 환산월세 "
            +
            station_map[
                "중앙"
            ].map(
                lambda x:
                    f"{x:.1f}만원"
            )
        )

        station_map[
            "tip_5"
        ] = (
            "500m 역세권"
        )


        station_data = pd.DataFrame(
            station_map.drop(
                columns="geometry",
                errors="ignore"
            )
        )


        # ----------------------------------------------------
        # 서울 자치구 경계
        # ----------------------------------------------------

        gu_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gu,
            filled=False,
            stroked=True,
            get_line_color=[
                80,
                80,
                80,
                80
            ],
            line_width_min_pixels=0.7,
            pickable=False
        )


        # ----------------------------------------------------
        # 역 마커
        # ----------------------------------------------------

        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=station_data,
            get_position=[
                "longitude",
                "latitude"
            ],
            get_fill_color="fill_color",
            get_line_color=[
                70,
                70,
                70,
                160
            ],
            get_radius="radius",
            radius_min_pixels=4,
            radius_max_pixels=14,
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True
        )


        # ----------------------------------------------------
        # 레이어 구성
        # ----------------------------------------------------

        layers = [
            gu_layer
        ]


        # 500m 버퍼 자동 표시
        if (
            buffer_map is not None
            and not buffer_map.empty
        ):

            buffer_layer = pdk.Layer(
                "GeoJsonLayer",
                data=buffer_map,

                # geometry 자체가 LineString
                filled=False,
                stroked=True,

                get_line_color=[
                    105,
                    105,
                    105,
                    90
                ],

                line_width_min_pixels=0.8,
                line_width_max_pixels=1.5,

                pickable=False
            )

            layers.append(
                buffer_layer
            )


        # 역 마커는 버퍼 위
        layers.append(
            station_layer
        )


        # LH는 가장 위
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


        # ----------------------------------------------------
        # 지도 설명
        # ----------------------------------------------------

        if selected_subway_lines:

            line_text = ", ".join(
                selected_subway_lines
            )

            st.caption(
                f"지도는 {line_text}의 역별 평균 환산월세와 "
                f"500m 역세권 범위를 표시합니다. "
                f"색상은 평균 환산월세의 5-95분위"
                f"({vmin:.1f}-{vmax:.1f}만원), "
                f"원의 크기는 거래건수를 나타냅니다."
            )

        else:

            st.caption(
                "지도는 서울 지하철역별 평균 환산월세와 "
                "500m 역세권 범위를 표시합니다. "
                f"색상은 평균 환산월세의 5-95분위"
                f"({vmin:.1f}-{vmax:.1f}만원), "
                "원의 크기는 거래건수를 나타냅니다."
            )


        if (
            buffer_map is None
            or buffer_map.empty
        ):

            st.caption(
                "현재 지도에 표시할 수 있는 "
                "500m 버퍼 geometry가 없어 "
                "버퍼 경계는 생략했습니다."
            )


        # ----------------------------------------------------
        # 역별 상세 통계
        # ----------------------------------------------------

        station_display = (
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
                    "station_name":
                        "역명",

                    "hoseon":
                        "노선",

                    "평균":
                        "평균 환산월세",

                    "중앙":
                        "중앙 환산월세",

                    "최저":
                        "최저 환산월세",

                    "최고":
                        "최고 환산월세"
                }
            )
        )


        with st.expander(
            "🚉 지하철역별 상세 통계 보기"
        ):

            st.dataframe(
                station_display,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # 전체 노선별 평균
        # ----------------------------------------------------

        st.markdown(
            "#### 🚇 노선별 평균 환산월세"
        )

        st.caption(
            "지도에서 선택한 노선과 관계없이 현재 주택유형·보증금·"
            "면적·연식·층수 조건에 해당하는 전체 노선을 비교합니다. "
            "같은 거래가 같은 노선의 여러 역 500m에 포함되는 경우 "
            "한 번만 계산합니다."
        )


        if not line_stats.empty:

            line_chart_df = (
                line_stats.copy()
            )

            line_chart_df[
                "노선색"
            ] = (
                line_chart_df[
                    "노선"
                ]
                .map(
                    LINE_COLOR_MAP
                )
                .fillna(
                    "#808080"
                )
            )

            line_color_domain = (
                line_chart_df[
                    "노선"
                ].tolist()
            )

            line_color_range = (
                line_chart_df[
                    "노선색"
                ].tolist()
            )


            line_chart = (
                alt.Chart(
                    line_chart_df
                )
                .mark_bar()
                .encode(

                    x=alt.X(
                        "평균:Q",
                        title="평균 환산월세 (만원)"
                    ),

                    y=alt.Y(
                        "노선:N",
                        sort="-x",
                        title="노선"
                    ),

                    color=alt.Color(
                        "노선:N",

                        scale=alt.Scale(
                            domain=line_color_domain,
                            range=line_color_range
                        ),

                        legend=None
                    ),

                    tooltip=[
                        alt.Tooltip(
                            "노선:N",
                            title="노선"
                        ),

                        alt.Tooltip(
                            "평균:Q",
                            title="평균 환산월세",
                            format=".1f"
                        ),

                        alt.Tooltip(
                            "중앙:Q",
                            title="중앙 환산월세",
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
                line_chart,
                use_container_width=True
            )


            line_display_df = (
                line_stats
                .rename(
                    columns={
                        "평균":
                            "평균 환산월세",

                        "중앙":
                            "중앙 환산월세"
                    }
                )
                .sort_values(
                    "평균 환산월세",
                    ascending=False
                )
            )


            with st.expander(
                "📋 노선별 상세 통계 보기"
            ):

                st.dataframe(
                    line_display_df,
                    use_container_width=True,
                    hide_index=True
                )


    # ========================================================
    # 23. LH 청년매입임대 결과
    # ========================================================

    if show_policy:

        st.divider()

        st.subheader(
            "🏠 조건에 맞는 LH 청년매입임대"
        )

        st.caption(
            "정책주택은 LH 공고의 전용면적을 사용하므로 "
            "민간 실거래의 임대면적과 면적 개념이 다릅니다. "
            "층수·주택유형 조건은 동일하게 적용합니다. "
            "공고 자료에 건축년도 정보가 없어 "
            "건물 연식 조건은 정책주택에 적용하지 않습니다."
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
                (
                    f"{policy_map['building_id'].nunique():,}"
                    "개"
                )
            )


            policy_metric2.metric(
                "조건에 맞는 공급호실",
                (
                    f"{len(policy_units_filtered):,}"
                    "호"
                )
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
                "지도에서 보라색 네모(■)는 "
                "조건에 맞는 LH 청년매입임대입니다."
            )


            if len(
                selected_base_deposits
            ) > 1:

                st.caption(
                    "기준 보증금을 여러 개 선택한 경우 "
                    "LH 위치와 공급호실은 표시하지만 "
                    "공식 전환월세는 계산하지 않습니다. "
                    "전환월세 비교가 필요하면 기준 보증금을 "
                    "하나만 선택해 주세요."
                )


            else:

                st.caption(
                    f"월세 비교값은 {policy_priority}의 공고상 "
                    "'기본 임대조건-임대료→보증금 최대전환' "
                    f"두 지점 사이를 선형 보간하여 보증금 "
                    f"{policy_base_deposit:,}만원 기준으로 계산합니다. "
                    "민간 거래에 사용하는 0.5% 환산식은 "
                    "정책주택에 적용하지 않습니다."
                )


            # ------------------------------------------------
            # 건물별 목록
            # ------------------------------------------------

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
                        "지오코딩주소":
                            "주소",

                        "조건공급호수":
                            "조건 공급호수",

                        "비교가능호수":
                            "월세 비교가능호수",

                        "전용면적_최소":
                            "전용면적 최소",

                        "전용면적_최대":
                            "전용면적 최대",

                        "정책월세_중앙":
                            "전환월세 중앙",

                        "정책월세_최저":
                            "전환월세 최저",

                        "정책월세_최고":
                            "전환월세 최고"
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


            # ------------------------------------------------
            # 호실별 목록
            # ------------------------------------------------

            if policy_priority == "청년 1순위":

                detail_prefix = (
                    "청년1순위"
                )

            else:

                detail_prefix = (
                    "청년23순위"
                )


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
                        "지오코딩주소":
                            "주소",

                        f"{detail_prefix}_기본보증금_만원":
                            "기본 보증금",

                        f"{detail_prefix}_기본월세_만원":
                            "기본 월세",

                        f"{detail_prefix}_최대전환보증금_만원":
                            "최대전환 보증금",

                        f"{detail_prefix}_최대전환월세_만원":
                            "최대전환 월세",

                        "정책월세_만원":
                            (
                                f"보증금 {policy_base_deposit:,}만원 전환월세"
                                if policy_base_deposit is not None
                                else "선택 보증금 전환월세"
                            )
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


    # ========================================================
    # 24. 서울 임대시장의 가격 구조
    # ========================================================

    st.divider()

    st.subheader(
        "📈 서울 임대시장의 가격 구조"
    )


    chart_col1, chart_col2 = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # 보증금 수준별 월세
    # --------------------------------------------------------

    deposit_stats = (
        make_deposit_stats(
            common_df,
            area_min,
            area_max
        )
    )


    with chart_col1:

        st.markdown(
            "#### 보증금 수준별 월세"
        )

        st.caption(
            f"임대면적 {area_min}-{area_max}㎡의 "
            "실제 월세 중앙값입니다."
        )

        deposit_chart = (
            alt.Chart(
                deposit_stats
            )
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


    # --------------------------------------------------------
    # 면적별 월세
    # --------------------------------------------------------

    area_stats = (
        make_area_stats(
            common_df,
            selected_base_deposits
        )
    )


    with chart_col2:

        st.markdown(
            "#### 면적별 월세"
        )

        st.caption(
            f"선택 보증금 {selected_deposit_text}의 각 구간을 "
            "해당 대표 보증금으로 환산한 월세 중앙값입니다."
        )


        if not area_stats.empty:

            area_chart = (
                alt.Chart(
                    area_stats
                )
                .mark_line(
                    point=True,
                    strokeWidth=2
                )
                .encode(

                    x=alt.X(
                        "면적대:N",

                        # 5-10이 가운데 들어가는 문제 방지
                        sort=AREA_BIN_ORDER,

                        title="임대면적 (㎡)"
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
