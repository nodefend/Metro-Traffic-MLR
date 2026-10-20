# PHÂN TÍCH VÀ DỰ ĐOÁN LƯU LƯỢNG GIAO THÔNG LIÊN TIỂU BANG METRO

## 1. Giới thiệu đề tài

Đề tài sử dụng bộ dữ liệu **Metro Interstate Traffic Volume** để phân tích đặc điểm lưu lượng giao thông, tìm hiểu mối liên hệ giữa lưu lượng giao thông với các yếu tố thời gian và thời tiết, đồng thời xây dựng mô hình dự đoán bằng phương pháp Hồi quy tuyến tính đa biến (Multiple Linear Regression).

Dự án kết hợp các phương pháp thống kê, trực quan hóa dữ liệu và học máy. Ngoài ra, nhóm xây dựng dashboard bằng Streamlit nhằm trình bày kết quả phân tích và minh họa khả năng dự đoán của mô hình.

## 2. Mục tiêu đề tài

- Khám phá đặc điểm và xu hướng lưu lượng giao thông theo giờ, ngày trong tuần và điều kiện thời tiết.
- Phân tích thống kê mô tả và phân phối xác suất của các biến.
- Thực hiện kiểm định giả thuyết để đánh giá sự khác biệt giữa các nhóm dữ liệu.
- Phân tích tương quan giữa lưu lượng giao thông và các biến định lượng.
- Xây dựng mô hình Hồi quy tuyến tính đa biến để dự đoán lưu lượng giao thông.
- So sánh hiệu quả mô hình với mô hình cơ sở (Baseline).
- Xây dựng dashboard trực quan để trình bày dữ liệu và thử nghiệm dự đoán.

## 3. Cấu trúc dự án

```text
Metro-Traffic-MLR/
├── notebooks/
│   ├── 01_EDA_and_Probability_Distributions.ipynb
│   ├── 02_Hypothesis_Correlation.ipynb
│   └── 03_Multiple_Linear_Regression.ipynb
├── dashboard/
│   ├── app.py
│   └── requirements.txt
├── outputs/
│   └── cleaned_traffic.csv
└── README.md
```

*Lưu ý: Đây là cấu trúc đề xuất. Tên thư mục và vị trí các file cần khớp với repository thực tế của nhóm.*

## 4. Nội dung các notebook

### Notebook 01: Khám phá dữ liệu và phân phối xác suất

Notebook tập trung vào việc tìm hiểu dữ liệu và các đặc điểm thống kê.

Các nội dung chính:
- Đọc và kiểm tra dữ liệu ban đầu.
- Làm sạch và tiền xử lý dữ liệu.
- Tính toán các thống kê mô tả.
- Trực quan hóa lưu lượng giao thông và các biến liên quan.
- Khảo sát phân phối của các biến được lựa chọn.

### Notebook 02: Kiểm định giả thuyết và phân tích tương quan

Notebook sử dụng các phương pháp thống kê để tìm hiểu sự khác biệt và mối liên hệ giữa các biến.

Các nội dung chính:
- Thực hiện kiểm định giả thuyết giữa các nhóm dữ liệu.
- Phân tích tương quan giữa các biến định lượng.
- Trực quan hóa và diễn giải kết quả thống kê.
- Đưa ra nhận xét dựa trên mức ý nghĩa thống kê được lựa chọn.

### Notebook 03: Hồi quy tuyến tính đa biến

Notebook xây dựng và đánh giá mô hình dự đoán lưu lượng giao thông.

Các nội dung chính:
- Xác định `traffic_volume` là biến mục tiêu.
- Lựa chọn các biến đầu vào về thời tiết và thời gian.
- Xử lý giá trị thiếu và mã hóa biến phân loại.
- Xây dựng quy trình tiền xử lý kết hợp mô hình.
- Chia dữ liệu theo thời gian với 80% tập huấn luyện và 20% tập kiểm tra.
- Huấn luyện mô hình Multiple Linear Regression và mô hình Baseline.
- Đánh giá bằng MAE, RMSE và R².
- Phân tích kết quả dự đoán, phần dư và hạn chế của mô hình.

## 5. Bộ dữ liệu

Đề tài sử dụng bộ dữ liệu **Metro Interstate Traffic Volume**.

Biến mục tiêu:
- `traffic_volume`: lưu lượng giao thông quan sát được.

Một số biến đầu vào chính:
- `temp`: nhiệt độ.
- `rain_1h`: lượng mưa trong một giờ.
- `snow_1h`: lượng tuyết trong một giờ.
- `clouds_all`: độ che phủ của mây.
- `holiday`: thông tin ngày lễ.
- `weather_main`: điều kiện thời tiết tổng quát.
- `date_time`: thời điểm ghi nhận dữ liệu, được sử dụng để tạo các đặc trưng thời gian như giờ, ngày trong tuần và tháng.

Dữ liệu CSV dùng cho dashboard cần có các cột mà chương trình yêu cầu. Nhóm có thể sử dụng file dữ liệu đã làm sạch `outputs/cleaned_traffic.csv` nếu file này được tạo từ notebook 01.

## 6. Mô hình học máy

Nhóm sử dụng **Hồi quy tuyến tính đa biến (Multiple Linear Regression)** để dự đoán lưu lượng giao thông dựa trên các yếu tố thời tiết và thời gian.

Quy trình xây dựng mô hình gồm:
- Điền giá trị thiếu của biến định lượng bằng trung vị.
- Điền giá trị thiếu của biến phân loại bằng giá trị xuất hiện nhiều nhất.
- Mã hóa biến phân loại bằng One-Hot Encoding.
- Chia dữ liệu theo trình tự thời gian để đánh giá mô hình trên giai đoạn dữ liệu về sau.

Mô hình được so sánh với Baseline sử dụng giá trị trung bình của tập huấn luyện.

Các chỉ số đánh giá:
- **MAE:** Sai số tuyệt đối trung bình giữa giá trị thực tế và dự đoán.
- **RMSE:** Căn bậc hai của sai số bình phương trung bình, nhạy hơn với các sai số lớn.
- **R²:** Hệ số xác định, thể hiện mức độ biến thiên của biến mục tiêu được mô hình giải thích trên tập đánh giá.

## 7. Dashboard trực quan

Dashboard được xây dựng bằng Streamlit, gồm bốn phần chính:

1. **Tổng quan:** Hiển thị thống kê cơ bản và lưu lượng giao thông theo giờ, ngày trong tuần.
2. **Khám phá dữ liệu và phân phối:** Hiển thị histogram, boxplot, scatter plot, thống kê mô tả và một số kiểm tra phân phối.
3. **Kiểm định và tương quan:** Trình bày kết quả kiểm định thống kê và ma trận tương quan.
4. **Dự đoán bằng MLR:** So sánh mô hình với Baseline, hiển thị MAE, RMSE, R², biểu đồ phần dư và cho phép nhập các điều kiện đầu vào để thử dự đoán.

## 8. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.11.

### Bước 1: Tải dự án

```bash
git clone https://github.com/USERNAME/Metro-Traffic-MLR.git
cd Metro-Traffic-MLR
```

Thay `USERNAME` bằng tên tài khoản GitHub của nhóm.

### Bước 2: Tạo môi trường ảo

Trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Trên macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt thư viện

Nếu file `requirements.txt` nằm trong thư mục `dashboard`:

```bash
pip install -r dashboard/requirements.txt
```

Nếu file nằm ở thư mục gốc:

```bash
pip install -r requirements.txt
```

## 9. Chạy dashboard

Đảm bảo file CSV nằm đúng đường dẫn mà `dashboard/app.py` sử dụng.

Tại thư mục gốc dự án, chạy:

```bash
streamlit run dashboard/app.py
```

Nếu file `app.py` nằm ở thư mục gốc, sử dụng:

```bash
streamlit run app.py
```

Sau đó mở địa chỉ Local URL được hiển thị trong Terminal để sử dụng dashboard.

## 10. Khả năng tái lập và hạn chế

- Kết quả phụ thuộc vào dữ liệu đầu vào và quy trình tiền xử lý.
- Việc chia dữ liệu theo thời gian giúp đánh giá khả năng dự đoán trên giai đoạn dữ liệu về sau.
- Mối tương quan hoặc sự khác biệt có ý nghĩa thống kê không mặc nhiên chứng minh quan hệ nhân quả.
- Mô hình hồi quy tuyến tính có thể chưa biểu diễn đầy đủ các mối quan hệ phi tuyến hoặc tương tác phức tạp.
- Các chỉ số đánh giá cần được diễn giải trong bối cảnh dữ liệu và giả định của mô hình.

## 11. Công nghệ sử dụng

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Streamlit
- Plotly

## 12. Kết luận

Dự án kết hợp phân tích thống kê, trực quan hóa dữ liệu và học máy để tìm hiểu đặc điểm lưu lượng giao thông và xây dựng mô hình dự đoán. Ba notebook trình bày quy trình phân tích chính, trong khi dashboard hỗ trợ trực quan hóa kết quả và minh họa khả năng ứng dụng của mô hình.

