import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np

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

# 4. 지하철 노선 목록
subway_reference = load_subway()

all_subway_lines = sorted({
    line.strip()
    for text in subway_reference["hoseon"].dropna()
    for line in str(text).split(",")
})

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

    deposit_options = {
        "1000만원 미만": (0, 1000, 500),
        "1000~3000만원 미만": (1000, 3000, 1000),
        "3000~5000만원 미만": (3000, 5000, 3000),
        "5000만원~1억원": (5000, 10000, 5000)
    }

    selected_deposit_label = st.selectbox(
        "보증금 구간",
        list(deposit_options.keys())
    )

    area_options = {
        "15㎡ 미만": (0, 15),
        "15~25㎡": (15, 25),
        "25~35㎡": (25, 35),
        "35㎡ 이상": (35, 9999)
    }

    selected_area_label = st.selectbox(
        "면적대",
        list(area_options.keys())
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
        "시각화 실행",
        type="primary",
        use_container_width=True
    )

# 6. 전월세 데이터 필터링
def filter_rent_data(
    df,
    house_type,
    dep_min,
    dep_max,
    base_deposit,
    area_min,
    area_max,
    building_age,
    floor
):
    # 월세만 사용
    df = df[
        df["전월세구분"] == "월세"
    ].copy()

    # 주택 유형
    if house_type != "전체":
        df = df[
            df["건물용도"] == house_type
        ]

    # 보증금
    df = df[
        (df["보증금(만원)"] >= dep_min) &
        (df["보증금(만원)"] < dep_max)
    ]

    # 면적
    df = df[
        (df["임대면적"] >= area_min) &
        (df["임대면적"] < area_max)
    ]

    # 건축년도
    if building_age == "신축 (2021년 이후)":
        df = df[
            df["건축년도"] >= 2021
        ]

    elif building_age == "2001년 이후":
        df = df[
            df["건축년도"] > 2000
        ]

    # 층수
    if floor == "저층 (1층 이하)":
        df = df[
            df["층"] <= 1
        ]

    # 보증금 기준 환산월세
    df["환산월세(만원)"] = (
        df["임대료(만원)"]
        + (df["보증금(만원)"] - base_deposit) * 0.005
    )

    return df

# 7. 공간 단위별 통계
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

# 8. 지도 색상
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

# 9. 서울 기본 지도 위치
SEOUL_VIEW = pdk.ViewState(
    latitude=37.5665,
    longitude=126.9780,
    zoom=10.3,
    pitch=0
)

# 10. 시각화 실행
if submit_button:
    dep_min, dep_max, base_deposit = (
        deposit_options[selected_deposit_label]
    )

    area_min, area_max = (
        area_options[selected_area_label]
    )

    # 선택 연도만 로드
    with st.spinner(
        f"{selected_year}년 거래 데이터를 불러오는 중..."
    ):
        df_raw = load_rent(selected_year)

    df = filter_rent_data(
        df_raw,
        house_type_selection,
        dep_min,
        dep_max,
        base_deposit,
        area_min,
        area_max,
        selected_age,
        selected_floor
    )

    if df.empty:
        st.warning(
            "선택한 조건에 해당하는 거래가 없습니다."
        )
        st.stop()

    # 11. 요약 통계
    st.subheader(
        f"📊 {selected_year}년 "
        f"{house_type_selection} "
        f"{spatial_unit} 환산월세"
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

    st.caption(
        f"보증금 {base_deposit:,}만원을 기준으로 "
        f"환산한 월세입니다."
    )

    # 12. 법정동별
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

        # 최소 표본 수
        dong_map = dong_map[
            dong_map["거래건수"] >= 5
        ].copy()

        if dong_map.empty:
            st.warning(
                "거래 5건 이상인 법정동이 없습니다."
            )
            st.stop()

        dong_map, vmin, vmax = add_map_color(
            dong_map,
            "평균"
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
            f"지도 색상은 법정동별 평균 환산월세의 "
            f"5~95분위({vmin:.1f}~{vmax:.1f}만원)를 "
            f"기준으로 표시하며, 거래 5건 미만 지역은 "
            f"제외했습니다."
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
                "평균",
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

    # 13. 500m 격자별
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
            grid_map,
            "평균"
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
            f"500m 격자별 평균 환산월세입니다. "
            f"색상은 5~95분위"
            f"({vmin:.1f}~{vmax:.1f}만원)를 기준으로 "
            f"표시하며, 거래 5건 미만 격자는 제외했습니다."
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
                "평균",
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

    # 14. 지하철역별
    elif spatial_unit == "지하철역별":
        rent_station = load_rent_station()
        subway = load_subway()
        gu = load_gu()

        # 현재 필터에 남은 거래와
        # 500m 역세권 관계 연결
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
                "선택한 조건에 해당하는 "
                "역세권 거래가 없습니다."
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

        # 선택 노선 필터
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

        # 최소 표본 수
        station_map = station_map[
            station_map["거래건수"] >= 5
        ].copy()

        if station_map.empty:
            st.warning(
                "선택한 조건에 해당하는 "
                "거래 5건 이상의 지하철역이 없습니다."
            )
            st.stop()

        station_map, vmin, vmax = add_map_color(
            station_map,
            "평균"
        )

        # 거래건수에 따라 원 크기 조절
        station_map["radius"] = (
            100
            + np.sqrt(
                station_map["거래건수"]
            ) * 15
        ).clip(
            lower=100,
            upper=400
        )

        # ScatterplotLayer에서는
        # shapely geometry가 필요 없으므로 제거
        station_data = pd.DataFrame(
            station_map.drop(
                columns="geometry",
                errors="ignore"
            )
        )

        # 자치구 경계
        gu_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gu,
            filled=False,
            stroked=True,
            get_line_color=[80, 80, 80, 80],
            line_width_min_pixels=0.7,
            pickable=False
        )

        # 지하철역
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

        # 노선을 선택한 경우 역명 표시
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
                f"원의 크기는 거래건수를 나타냅니다. "
                f"거래 5건 미만 역은 제외했습니다."
            )

        else:
            st.caption(
                f"서울 지하철역의 500m 역세권 기준입니다. "
                f"색상은 평균 환산월세의 5~95분위"
                f"({vmin:.1f}~{vmax:.1f}만원), "
                f"원의 크기는 거래건수를 나타냅니다. "
                f"거래 5건 미만 역은 제외했습니다."
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
                "평균",
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

else:
    st.info(
        "왼쪽 사이드바에서 검색 조건을 선택한 후 "
        "'시각화 실행' 버튼을 눌러주세요."
    )
