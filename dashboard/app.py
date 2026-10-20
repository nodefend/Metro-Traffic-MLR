import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Metro Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Metro Interstate Traffic Volume")
st.caption("Dashboard phân tích dữ liệu, kiểm định thống kê và dự đoán lưu lượng giao thông bằng Multiple Linear Regression.")

REQUIRED = [
    "date_time", "traffic_volume", "temp", "rain_1h",
    "snow_1h", "clouds_all", "holiday", "weather_main"
]
NUMERIC_FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "holiday", "weather_main"]
TARGET = "traffic_volume"

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    return df

def prepare_data(raw):
    df = raw.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing))
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    for c in NUMERIC_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    df = df.dropna(subset=["date_time", TARGET])
    # Keep the same key cleaning principle as the group's notebooks.
    if "temp" in df.columns:
        df.loc[df["temp"] <= 0, "temp"] = np.nan
    for c in ["rain_1h", "snow_1h", "clouds_all"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["day_type"] = np.where(df["day_of_week"] < 5, "Weekday", "Weekend")
    df["temp_c"] = df["temp"] - 273.15
    df["holiday"] = df["holiday"].fillna("None").astype(str)
    df["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)
    df = df.sort_values("date_time").reset_index(drop=True)
    return df

def metrics(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R²": r2_score(y, pred)
    }

def fit_model(df):
    data = df.dropna(subset=[TARGET, "date_time"]).sort_values("date_time").copy()
    if len(data) < 30:
        raise ValueError("Cần ít nhất 30 dòng dữ liệu hợp lệ để huấn luyện và đánh giá.")
    split = int(len(data) * 0.8)
    split = min(max(split, 1), len(data)-1)
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
    numeric = [c for c in NUMERIC_FEATURES if c in data.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in data.columns]
    features = numeric + categorical
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]), categorical)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    baseline = DummyRegressor(strategy="mean")
    model.fit(train[features], train[TARGET])
    baseline.fit(np.zeros((len(train), 1)), train[TARGET])
    pred = model.predict(test[features])
    base_pred = baseline.predict(np.zeros((len(test), 1)))
    result = test[["date_time", TARGET]].copy().rename(columns={TARGET: "actual"})
    result["predicted"] = pred
    result["residual"] = result["actual"] - result["predicted"]
    return model, features, train, test, result, metrics(test[TARGET], pred), metrics(test[TARGET], base_pred)

with st.sidebar:
    st.header("Dữ liệu")
    st.caption("Dashboard tự tìm CSV trong thư mục dự án. Chỉ cần tải file nếu muốn đổi sang bộ dữ liệu khác.")
    uploaded = st.file_uploader("Tùy chọn: thay đổi CSV", type=["csv"])
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent

    candidate_paths = [
        PROJECT_DIR / "outputs" / "cleaned_traffic.csv",
        BASE_DIR / "outputs" / "cleaned_traffic.csv",
        PROJECT_DIR / "cleaned_traffic.csv",
    ]

    candidate = next((p for p in candidate_paths if p.exists()), None)

    if candidate:
        raw = pd.read_csv(candidate)
        data_source = str(candidate)
    else:
        raw = None
        data_source = None
    if uploaded is not None:
        raw = load_data(uploaded.getvalue(), uploaded.name)
        data_source = uploaded.name
    else:
        candidate = next((p for p in candidate_paths if p.exists()), None)
        if candidate:
            raw = pd.read_csv(candidate)
            data_source = str(candidate)
        else:
            raw = None
            data_source = None
    st.divider()
    page = st.radio(
        "Điều hướng",
        ["Tổng quan", "EDA & Phân phối", "Kiểm định & Tương quan", "Dự đoán MLR"],
        index=0
    )

if raw is None:
    st.error("Dashboard chưa tìm thấy file CSV trong thư mục dự án.")
    st.markdown("""
    **Cách sửa một lần để dashboard tự phân tích khi mở**
    1. Tạo thư mục `outputs` ngay cạnh `app.py`.
    2. Đặt file dữ liệu đã làm sạch từ notebook 01 vào đó, tên `cleaned_traffic.csv`.
    3. Chạy lại `streamlit run app.py`.

    Khi CSV đã nằm đúng vị trí, dashboard sẽ tự đọc dữ liệu và hiển thị các trang phân tích. Mục tải CSV phía trên chỉ là tùy chọn khi muốn đổi dữ liệu.
    """)
    st.stop()

try:
    df = prepare_data(raw)
except Exception as e:
    st.error(f"Không thể đọc dữ liệu: {e}")
    st.info("CSV cần có các cột: date_time, traffic_volume, temp, rain_1h, snow_1h, clouds_all, holiday, weather_main.")
    st.stop()

with st.sidebar:
    st.success(f"Đã tải: {data_source or 'CSV tải lên'}")
    st.metric("Số dòng hợp lệ", f"{len(df):,}")
    min_date, max_date = df["date_time"].min().date(), df["date_time"].max().date()
    date_range = st.date_input("Khoảng thời gian", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        view = df[(df["date_time"].dt.date >= date_range[0]) & (df["date_time"].dt.date <= date_range[1])].copy()
    else:
        view = df.copy()

if view.empty:
    st.warning("Không có dữ liệu trong khoảng thời gian được chọn.")
    st.stop()

if page == "Tổng quan":
    st.subheader("Tổng quan dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số quan sát", f"{len(view):,}")
    c2.metric("Lưu lượng trung bình", f"{view[TARGET].mean():,.0f} xe/giờ")
    c3.metric("Lưu lượng trung vị", f"{view[TARGET].median():,.0f} xe/giờ")
    c4.metric("Số thời điểm", f"{view['date_time'].nunique():,}")
    left, right = st.columns(2)
    hourly = view.groupby("hour", as_index=False)[TARGET].mean()
    fig = px.line(hourly, x="hour", y=TARGET, markers=True, title="Lưu lượng trung bình theo giờ")
    fig.update_layout(xaxis_title="Giờ trong ngày", yaxis_title="Xe/giờ")
    left.plotly_chart(fig, use_container_width=True)
    daily = view.groupby("day_of_week", as_index=False)[TARGET].mean()
    days = {0:"Thứ Hai",1:"Thứ Ba",2:"Thứ Tư",3:"Thứ Năm",4:"Thứ Sáu",5:"Thứ Bảy",6:"Chủ nhật"}
    daily["day"] = daily["day_of_week"].map(days)
    fig = px.bar(daily, x="day", y=TARGET, title="Lưu lượng trung bình theo ngày trong tuần")
    fig.update_layout(xaxis_title="", yaxis_title="Xe/giờ")
    right.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Dữ liệu sau xử lý")
    st.dataframe(view.head(100), use_container_width=True, hide_index=True)
    st.download_button("Tải CSV dữ liệu đang xem", view.to_csv(index=False).encode("utf-8-sig"), "traffic_data_view.csv", "text/csv")

elif page == "EDA & Phân phối":
    st.subheader("Khám phá dữ liệu và phân phối")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(view, x=TARGET, nbins=40, marginal="box", title="Phân phối lưu lượng giao thông")
        fig.update_layout(xaxis_title="Xe/giờ", yaxis_title="Tần suất")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        temp_plot = view[["temp_c", TARGET]].dropna()
        if len(temp_plot) > 5000:
            temp_plot = temp_plot.sample(5000, random_state=42)
        fig = px.scatter(temp_plot, x="temp_c", y=TARGET, opacity=0.35, title="Nhiệt độ và lưu lượng giao thông")
        fig.update_layout(xaxis_title="Nhiệt độ (°C)", yaxis_title="Xe/giờ")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Boxplot các biến định lượng")
    chosen = st.multiselect("Chọn biến", ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all"], default=["traffic_volume", "temp_c", "clouds_all"])
    if chosen:
        long_df = view[chosen].melt(var_name="Biến", value_name="Giá trị").dropna()
        fig = px.box(long_df, x="Biến", y="Giá trị", points=False, title="Boxplot")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Thống kê mô tả")
    st.dataframe(view[["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all"]].describe().T.round(3), use_container_width=True)
    st.markdown("#### Kiểm tra phân phối chuẩn (Shapiro–Wilk)")
    st.caption("Do giới hạn của kiểm định, lấy mẫu tối đa 5.000 quan sát; p-value nhạy với kích thước mẫu.")
    normal_rows = []
    for col in ["traffic_volume", "temp_c"]:
        s = view[col].dropna()
        if len(s) >= 3:
            sample = s.sample(min(len(s), 5000), random_state=42) if len(s) > 5000 else s
            stat, pval = stats.shapiro(sample)
            normal_rows.append({"Biến": col, "Số mẫu kiểm định": len(sample), "W": stat, "p-value": pval,
                                "Kết luận (α=0,05)": "Bác bỏ H₀: không phân phối chuẩn" if pval < 0.05 else "Chưa đủ bằng chứng bác bỏ H₀"})
    if normal_rows:
        st.dataframe(pd.DataFrame(normal_rows).round(6), use_container_width=True)
    traffic = view[TARGET].dropna()
    if len(traffic) and traffic.mean() != 0:
        st.metric("Tỷ số phương sai / trung bình của traffic_volume", f"{traffic.var() / traffic.mean():,.2f}")
        st.caption("Với phân phối Poisson đơn giản, phương sai kỳ vọng xấp xỉ trung bình. Tỷ số này chỉ là đánh giá sơ bộ, không phải kiểm định Poisson đầy đủ.")

elif page == "Kiểm định & Tương quan":
    st.subheader("Kiểm định giả thuyết và phân tích tương quan")
    st.caption("Các kết luận dùng mức ý nghĩa α = 0,05. Kết quả là mối liên hệ thống kê, không tự động chứng minh quan hệ nhân quả.")
    daily = view.copy()
    daily["date"] = daily["date_time"].dt.date
    daily["day_of_week_name"] = daily["date_time"].dt.day_name()
    daily_daily = daily.groupby("date").agg(daily_mean=(TARGET, "mean"), n_hours=(TARGET, "count")).reset_index()
    complete_dates = set(daily_daily.loc[daily_daily["n_hours"] == 24, "date"])
    daily24 = daily[daily["date"].isin(complete_dates)].groupby("date").agg(
        daily_mean=(TARGET, "mean"),
        day_of_week=("day_of_week", "first"),
        day_type=("day_type", "first")
    ).reset_index()
    if len(daily24) >= 4:
        weekday = daily24.loc[daily24["day_type"] == "Weekday", "daily_mean"].dropna()
        weekend = daily24.loc[daily24["day_type"] == "Weekend", "daily_mean"].dropna()
        if len(weekday) >= 2 and len(weekend) >= 2:
            tstat, pval = stats.ttest_ind(weekday, weekend, equal_var=False)
            diff = weekday.mean() - weekend.mean()
            c1,c2,c3 = st.columns(3)
            c1.metric("TB ngày thường", f"{weekday.mean():,.1f}")
            c2.metric("TB cuối tuần", f"{weekend.mean():,.1f}")
            c3.metric("Chênh lệch", f"{diff:,.1f} xe/giờ")
            st.write(f"**Welch t-test:** t = {tstat:.3f}, p-value = {pval:.4g}. " +
                     ("Bác bỏ H₀: có bằng chứng về sự khác biệt trung bình." if pval < 0.05 else "Chưa đủ bằng chứng về sự khác biệt trung bình."))
        else:
            st.info("Không đủ ngày hoàn chỉnh cho cả hai nhóm ngày thường và cuối tuần trong khoảng thời gian đang chọn.")
    else:
        st.info("Cần có ít nhất vài ngày đủ 24 bản ghi để thực hiện kiểm định theo ngày.")

    st.markdown("#### Tương quan Pearson và Spearman")
    corr_cols = ["temp_c", "rain_1h", "snow_1h", "clouds_all", TARGET]
    corr_method = st.radio("Phương pháp", ["Pearson", "Spearman"], horizontal=True)
    corr = view[corr_cols].corr(method=corr_method.lower())
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Ma trận tương quan {corr_method}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(corr.round(3), use_container_width=True)

    st.markdown("#### Welch ANOVA giữa các ngày trong tuần")
    day_names = {0:"Thứ Hai",1:"Thứ Ba",2:"Thứ Tư",3:"Thứ Năm",4:"Thứ Sáu",5:"Thứ Bảy",6:"Chủ nhật"}
    groups = [daily24.loc[daily24["day_of_week"] == d, "daily_mean"].dropna().to_numpy() for d in range(7)] if len(daily24) else []
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        try:
            from statsmodels.stats.oneway import anova_oneway
            all_groups = [daily24.loc[daily24["day_of_week"] == d, "daily_mean"].dropna().to_numpy() for d in range(7)]
            all_groups = [g for g in all_groups if len(g) >= 2]
            res = anova_oneway(all_groups, use_var="unequal")
            st.write(f"Welch ANOVA: statistic = {float(res.statistic):.3f}, p-value = {float(res.pvalue):.4g}.")
            st.write("Kết luận: " + ("có ít nhất một nhóm ngày có trung bình khác biệt thống kê (α=0,05)." if res.pvalue < 0.05 else "chưa đủ bằng chứng cho thấy trung bình các ngày khác nhau."))
        except Exception as e:
            st.info(f"Không thể tính Welch ANOVA: {e}")
    else:
        st.info("Không đủ ngày hoàn chỉnh ở nhiều nhóm thứ để tính Welch ANOVA.")

    st.markdown("#### Liên hệ giữa nhóm thời tiết và mức lưu lượng")
    st.caption("High/Low được xác định tương đối theo trung vị trong khoảng dữ liệu đang xem, không phải nhãn ùn tắc.")
    weather_daily = view.copy()
    weather_daily["date"] = weather_daily["date_time"].dt.date
    weather_daily["weather_group"] = weather_daily["weather_main"].replace("", "Unknown").fillna("Unknown")
    weather_daily = weather_daily.groupby("date").agg(
        daily_mean=(TARGET, "mean"),
        weather_group=("weather_group", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    ).reset_index()
    if len(weather_daily) >= 10 and weather_daily["weather_group"].nunique() >= 2:
        threshold = weather_daily["daily_mean"].median()
        weather_daily["traffic_level"] = np.where(weather_daily["daily_mean"] >= threshold, "High", "Low")
        tab = pd.crosstab(weather_daily["weather_group"], weather_daily["traffic_level"])
        if tab.shape[0] >= 2 and tab.shape[1] == 2:
            chi2, pval, dof, expected = stats.chi2_contingency(tab)
            st.dataframe(tab, use_container_width=True)
            st.write(f"Chi-square: χ² = {chi2:.3f}, df = {dof}, p-value = {pval:.4g}.")
            st.write("Kết luận: " + ("có bằng chứng về mối liên hệ giữa nhóm thời tiết và mức lưu lượng tương đối." if pval < 0.05 else "chưa đủ bằng chứng về mối liên hệ thống kê."))
        else:
            st.info("Bảng chéo chưa đủ nhóm để thực hiện kiểm định.")
    else:
        st.info("Không đủ dữ liệu để kiểm định Chi-square trong khoảng đang chọn.")

elif page == "Dự đoán MLR":
    st.subheader("Mô hình Multiple Linear Regression")
    st.write("Mô hình được huấn luyện ngay trên dữ liệu đang tải. Dữ liệu được sắp xếp theo thời gian, dùng 80% giai đoạn đầu để train và 20% giai đoạn cuối để test.")
    with st.spinner("Đang huấn luyện mô hình và tính chỉ số..."):
        try:
            model, features, train, test, result, model_m, base_m = fit_model(df)
        except Exception as e:
            st.error(f"Không thể huấn luyện mô hình: {e}")
            st.stop()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE", f"{model_m['MAE']:,.2f}")
    c2.metric("RMSE", f"{model_m['RMSE']:,.2f}")
    c3.metric("R²", f"{model_m['R²']:.4f}")
    c4.metric("Số dòng test", f"{len(test):,}")
    compare = pd.DataFrame([
        {"Mô hình":"Baseline (mean)", **base_m},
        {"Mô hình":"Multiple Linear Regression", **model_m}
    ])
    st.markdown("#### So sánh với Baseline")
    st.dataframe(compare.round(3), use_container_width=True, hide_index=True)
    improvement = (base_m["MAE"] - model_m["MAE"]) / base_m["MAE"] * 100 if base_m["MAE"] else np.nan
    st.write(f"MAE cải thiện so với Baseline: **{improvement:.2f}%**.")
    left,right = st.columns(2)
    fig = px.scatter(result, x="actual", y="predicted", opacity=0.35, title="Giá trị thực tế và dự đoán")
    lo = min(result["actual"].min(), result["predicted"].min())
    hi = max(result["actual"].max(), result["predicted"].max())
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Dự đoán hoàn hảo", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Thực tế (xe/giờ)", yaxis_title="Dự đoán (xe/giờ)")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(result, x="predicted", y="residual", opacity=0.35, title="Biểu đồ phần dư")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(xaxis_title="Dự đoán (xe/giờ)", yaxis_title="Phần dư = Thực tế − Dự đoán")
    right.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    sample = result.head(300)
    fig.add_trace(go.Scatter(x=sample["date_time"], y=sample["actual"], mode="lines", name="Thực tế"))
    fig.add_trace(go.Scatter(x=sample["date_time"], y=sample["predicted"], mode="lines", name="Dự đoán"))
    fig.update_layout(title="Thực tế và dự đoán theo thời gian (tối đa 300 quan sát test)", xaxis_title="Thời gian", yaxis_title="Xe/giờ")
    st.plotly_chart(fig, use_container_width=True)
    result["absolute_error"] = result["residual"].abs()
    st.markdown("#### Sai số theo nhóm dự đoán")
    try:
        result["prediction_group"] = pd.qcut(result["predicted"], q=5, duplicates="drop")
        err = result.groupby("prediction_group", observed=True).agg(
            actual_mean=("actual","mean"), predicted_mean=("predicted","mean"),
            MAE=("absolute_error","mean"), mean_residual=("residual","mean"), count=("residual","size")
        ).reset_index()
        err["prediction_group"] = err["prediction_group"].astype(str)
        st.dataframe(err.round(2), use_container_width=True, hide_index=True)
    except Exception:
        st.info("Không thể chia nhóm dự đoán cho dữ liệu này.")
    st.markdown("#### Thử dự đoán với đầu vào mới")
    st.caption("Đầu vào phải nằm trong phạm vi dữ liệu đã có. Đây là ước lượng của mô hình, không phải số đo giao thông thực tế.")
    with st.form("predict_form"):
        a,b,c = st.columns(3)
        with a:
            input_date = st.date_input("Ngày", value=pd.Timestamp(df["date_time"].max()).date())
            input_hour = st.slider("Giờ trong ngày", 0, 23, 8)
            temp_c = st.number_input("Nhiệt độ (°C)", min_value=-50.0, max_value=60.0, value=float(np.nanmedian(df["temp_c"])), step=1.0)
        with b:
            rain = st.number_input("Lượng mưa (mm/giờ)", min_value=0.0, max_value=1000.0, value=float(np.nanmedian(df["rain_1h"].clip(lower=0))), step=0.1)
            snow = st.number_input("Lượng tuyết (mm/giờ)", min_value=0.0, max_value=1000.0, value=float(np.nanmedian(df["snow_1h"].clip(lower=0))), step=0.1)
            clouds = st.slider("Độ che phủ mây (%)", 0, 100, int(np.nanmedian(df["clouds_all"].clip(lower=0, upper=100))))
        with c:
            holiday_options = sorted(df["holiday"].astype(str).fillna("None").unique().tolist())
            weather_options = sorted(df["weather_main"].astype(str).fillna("Unknown").unique().tolist())
            holiday = st.selectbox("Ngày lễ", holiday_options)
            weather = st.selectbox("Thời tiết", weather_options)
        submitted = st.form_submit_button("Dự đoán lưu lượng")
    if submitted:
        dt = pd.Timestamp(input_date)
        day = dt.dayofweek
        row = {
            "temp": temp_c + 273.15,
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "hour": input_hour,
            "day_of_week": day,
            "month": dt.month,
            "holiday": holiday,
            "weather_main": weather
        }
        new = pd.DataFrame([row])[features]
        prediction = float(model.predict(new)[0])
        st.metric("Lưu lượng dự đoán", f"{max(0, prediction):,.0f} xe/giờ" if prediction >= 0 else f"{prediction:,.0f} xe/giờ (dự đoán âm)")
        if prediction < 0:
            st.warning("Mô hình tuyến tính cho kết quả âm. Lưu lượng thực tế không thể âm; đây là hạn chế của mô hình. Không tự động thay đổi kết quả đánh giá chính.")
    st.markdown("#### Tải kết quả dự đoán trên tập test")
    st.download_button("Tải CSV kết quả test", result.to_csv(index=False).encode("utf-8-sig"), "mlr_test_predictions.csv", "text/csv")

st.divider()
st.caption("Nguồn: bộ dữ liệu Metro Interstate Traffic Volume và quy trình phân tích trong 3 notebook của nhóm. Kết quả trên dashboard được tính lại từ CSV đang tải; có thể khác nhẹ nếu dữ liệu đầu vào hoặc cách làm sạch khác.")
