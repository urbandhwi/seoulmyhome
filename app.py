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
    "월세 수준을 법정동·500m 격자·지하철역별로 비교합니다."
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

# 4. 기준값 설정
subway_reference = load_subway()

all_subway_lines = sorted({
    line.strip()
    for text in subway_reference["hoseon"].dropna()
    for line in str(text).split(",")
})

# 보증금 대표값
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

    return base_deposit, base_deposit + 1000

# 5. 검색 조건
with st.sidebar.form("search_form"):
    st.header("🔍 검색 조건 설정")

    house_type_selection = st.radio(
        "주택 유형",
        ["전체", "연립다세대", "오피스텔"]
    )

    spatial_unit = st.radio(
        "시각화 단위",
        ["법정동별", "격자별", "지하철역별"]
    )

    selected_subway_lines = st.multiselect(
        "지하철 노선",
        options=all_subway_lines,
        default=[],
        help=(
            "지하철역별 시각화에 적용됩니다. "
            "선택하지 않으면 전체 역을 표시합니다."
        )
    )

    selected_year = st.selectbox(
        "연도",
        [2025, 2024, 2023]
    )

    selected_deposit_label = st.selectbox(
        "기준 보증금",
        list(DEPOSIT_OPTIONS.keys()),
        index=2,
        help=(
            "선택한 보증금 구간의 거래를 대표 보증금으로 "
            "환산하여 비교합니다."
        )
    )

    area_min, area_max = st.slider(
        "전용면적 (㎡)",
        min_value=5,
        max_value=85,
        value=(15, 30),
        step=1
    )

    st.caption(
        f"선택 면적: {area_min}㎡ ~ {area_max}㎡"
    )

    selected_age = st.selectbox(
        "건물 연식",
        [
            "전체",
            "신축 (2021년 이후)",
            "2001년 이후"
        ]
    )

    selected_floor = st.selectbox(
        "층수",
        [
            "전체",
            "저층 (1층 이하)"
        ]
    )

    submit_button = st.form_submit_button(
        "우리집 찾기",
        type="primary",
        use_container_width=True
    )

# 6. 공통 조건 필터
def filter_common_data(
    df,
    house_type,
    building_age,
    floor
):
    df = df[
        df["전월세구분"] == "월세"
    ].copy()

    if house_type != "전체":
        df = df[
            df["건물용도"] == house_type
        ]

    if building_age == "신축 (2021년 이후)":
        df = df[
            df["건축년도"] >= 2021
        ]

    elif building_age == "2001년 이후":
        df = df[
            df["건축년도"] > 2000
        ]

    if floor == "저층 (1층 이하)":
        df = df[
            df["층"] <= 1
        ]

    return df

# 7. 지도용 거래 필터
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

# 8. 공간 단위별 통계
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

# 9. 지도 색상
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

# 10. 보증금별 월세 통계
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

    order = list(DEPOSIT_OPTIONS.keys())

    chart_df["보증금기준"] = pd.Categorical(
        chart_df["보증금기준"],
        categories=order,
        ordered=True
    )

    stats = (
        chart_df.dropna(subset=["보증금기준"])
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

# 11. 면적별 월세 통계
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
        f"{start}~{start + 5}"
        for start in bins[:-1]
    ]

    chart_df["면적대"] = pd.cut(
        chart_df["임대면적"],
        bins=bins,
        labels=labels,
        right=False
    )

    stats = (
        chart_df.dropna(subset=["면적대"])
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

# 12. 서울 기본 지도
SEOUL_VIEW = pdk.ViewState(
    latitude=37.5665,
    longitude=126.9780,
    zoom=10.3,
    pitch=0
)

# 13. 실행
if submit_button:
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

    # 주택유형·연식·층 조건
    common_df = filter_common_data(
        df_raw,
        house_type_selection,
        selected_age,
        selected_floor
    )

    # 보증금·면적 조건
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

    # 14. 선택 조건 요약
    st.subheader(
        f"📊 {selected_year}년 "
        f"{house_type_selection} "
        f"{spatial_unit}"
    )

    st.markdown(
        f"**기준 보증금 {selected_deposit_label} · "
        f"전용면적 {area_min}~{area_max}㎡**"
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
            "실제 보증금 1~999만원 거래를 "
            "보증금 500만원 기준 월세로 환산했습니다."
        )

    else:
        st.caption(
            f"실제 보증금 {dep_min:,}~{dep_max - 1:,}만원 "
            f"거래를 보증금 {base_deposit:,}만원 기준 "
            f"월세로 환산했습니다."
        )

    # 15. 법정동별
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
            dong_map["거래건수"] >= 5
        ].copy()

        if dong_map.empty:
            st.warning(
                "거래 5건 이상인 법정동이 없습니다."
            )
            st.stop()

        dong_map, vmin, vmax = add_map_color(
            dong_map
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

        tooltip = {
            "html": """
                <b>{EMD_NM}</b><br/>
                거래건수: {거래건수}건<br/>
                평균 환산월세: {평균}만원<br/>
                중앙 환산월세: {중앙}만원<br/>
                최저 환산월세: {최저}만원<br/>
                최고 환산월세: {최고}만원
            """
        }

        deck = pdk.Deck(
            layers=[dong_layer],
            initial_view_state=SEOUL_VIEW,
            tooltip=tooltip,
            map_provider="carto",
            map_style="light"
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

        st.caption(
            f"색상은 법정동별 평균 환산월세의 "
            f"5~95분위({vmin:.1f}~{vmax:.1f}만원)를 "
            f"기준으로 하며 거래 5건 미만 지역은 제외합니다."
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
            .sort_values("평균", ascending=False)
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

    # 16. 격자별
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
            grid_map["거래건수"] >= 5
        ].copy()

        if grid_map.empty:
            st.warning(
                "거래 5건 이상인 격자가 없습니다."
            )
            st.stop()

        grid_map, vmin, vmax = add_map_color(
            grid_map
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

        tooltip = {
            "html": """
                <b>500m 격자 {grid_id}</b><br/>
                거래건수: {거래건수}건<br/>
                평균 환산월세: {평균}만원<br/>
                중앙 환산월세: {중앙}만원<br/>
                최저 환산월세: {최저}만원<br/>
                최고 환산월세: {최고}만원
            """
        }

        deck = pdk.Deck(
            layers=[grid_layer],
            initial_view_state=SEOUL_VIEW,
            tooltip=tooltip,
            map_provider="carto",
            map_style="light"
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

        st.caption(
            f"색상은 500m 격자별 평균 환산월세의 "
            f"5~95분위({vmin:.1f}~{vmax:.1f}만원)를 "
            f"기준으로 하며 거래 5건 미만 격자는 제외합니다."
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
            .sort_values("평균", ascending=False)
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

    # 17. 지하철역별
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
            station_map["거래건수"] >= 5
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

        tooltip = {
            "html": """
                <b>{station_name}</b><br/>
                {hoseon}<br/>
                거래건수: {거래건수}건<br/>
                평균 환산월세: {평균}만원<br/>
                중앙 환산월세: {중앙}만원<br/>
                최저 환산월세: {최저}만원<br/>
                최고 환산월세: {최고}만원
            """
        }

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=SEOUL_VIEW,
            tooltip=tooltip,
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
                f"색상은 평균 환산월세의 5~95분위"
                f"({vmin:.1f}~{vmax:.1f}만원), "
                f"원의 크기는 거래건수를 나타냅니다."
            )

        else:
            st.caption(
                f"서울 지하철역의 500m 역세권 기준입니다. "
                f"색상은 평균 환산월세의 5~95분위"
                f"({vmin:.1f}~{vmax:.1f}만원), "
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
            .sort_values("평균", ascending=False)
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

    # 18. 서울 임대시장 가격 구조
    st.divider()
    st.subheader("📈 서울 임대시장의 가격 구조")

    chart_col1, chart_col2 = st.columns(2)

    # 보증금별 실제 월세
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
            f"전용면적 {area_min}~{area_max}㎡의 "
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
                    sort=list(DEPOSIT_OPTIONS.keys()),
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

    # 면적별 환산월세
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
        "왼쪽 사이드바에서 검색 조건을 선택한 후 "
        "'우리집 찾기' 버튼을 눌러주세요."
    )
