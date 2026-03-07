import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Báo cáo phân cụm hội viên bể bơi", layout="wide")
st.markdown("""
<style>

/* bảng đầu */
table {
    border-collapse: collapse;
    width: 100%;
    font-size:16px;
}

/* header */
thead tr th {
    background-color:#669999 !important;
    color:#ffffff !important;
    font-weight:bold !important;
    text-align:center;
}

/* hàng lẻ */
tbody tr:nth-child(odd) {
    background-color:#bcd3d9;
}

/* hàng chẵn */
tbody tr:nth-child(even) {
    background-color:#ffffff;
}

/* border */
td, th {
    border:1px solid #c0c0c0;
    padding:6px;
}

/* cột tên đậm */
tbody tr td:first-child {
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Báo cáo phân cụm hội viên bể bơi")

# =========================
# Sidebar
# =========================

st.sidebar.markdown(
"""
<p style='font-size:22px; color:#000000; font-weight:bold;'>
Trường ĐH Công nghệ Kỹ thuật TP.HCM
</p>
""",
unsafe_allow_html=True
)
st.sidebar.subheader("Môn học: BIG DATA")
st.sidebar.write("Giảng viên: TS. Hồ Nhựt Minh")
st.sidebar.write("Sinh viên: Nguyễn Thị Xuân Tâm - 23126036")

st.sidebar.markdown("---")

st.sidebar.subheader("Từ yêu cầu đề bài:")
st.sidebar.write("random_state=42, n_init=10")

st.sidebar.subheader("⚙️ Thiết lập phân cụm")

chosen_k = st.sidebar.slider("Chọn số cụm (k)", 2, 8, 3)

st.sidebar.markdown("---")



# =========================
# 1. Dữ liệu
# =========================

@st.cache_data
def load_data():
    data = {
        'Tên': ['An', 'Bình', 'Cường', 'Dung', 'Em', 'Phong', 'Giang', 'Hà',
                'Khánh', 'Linh', 'Minh', 'Ngọc', 'Phúc', 'Quỳnh', 'Sơn'],
        'Số buổi bơi/tuần': [2, 5, 7, 1, 4, 8, 3, 6, 9, 2, 5, 7, 3, 6, 10],
        'Thời gian TB (phút)': [30, 60, 75, 20, 45, 90, 35, 65, 85, 25, 55, 80, 40, 70, 95],
        'Số lớp tham gia/tuần': [0, 2, 3, 0, 1, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5]
    }
    return pd.DataFrame(data)

df = load_data()

st.subheader("📋 Dữ liệu hội viên")
st.table(df)

# =========================
# 2. Chuẩn hóa dữ liệu
# =========================

numeric_cols = ['Số buổi bơi/tuần', 'Thời gian TB (phút)', 'Số lớp tham gia/tuần']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

st.subheader("🔄 Dữ liệu sau khi chuẩn hóa (StandardScaler)")
st.dataframe(df_scaled)

# =========================
# 3. Elbow Method
# =========================

st.subheader("📉 Elbow Method")

inertia_values = []
k_range = range(1, 9)

fig, ax = plt.subplots()

for k in k_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_tmp.fit(df_scaled)
    inertia_values.append(kmeans_tmp.inertia_)

ax.plot(k_range, inertia_values, marker='o')

ax.set_title("Elbow Method")
ax.set_xlabel("Số cụm (k)")
ax.set_ylabel("Inertia")
ax.grid(True)

st.pyplot(fig)

st.info("👉 Điểm gãy xuất hiện tại k = 3 → chọn k = 3 cho mô hình phân cụm.")

# =========================
# 4. K-Means
# =========================

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
kmeans.fit(df_scaled)

df['Cluster'] = kmeans.labels_

st.subheader("🏷️ Kết quả phân cụm")

st.dataframe(df[['Tên', 'Cluster']], use_container_width=True)

# =========================
# 5. Scatter Plot
# =========================

st.subheader("🎨 Biểu đồ phân cụm")

col_x = st.sidebar.selectbox("Chọn trục X", numeric_cols, index=0)
col_y = st.sidebar.selectbox("Chọn trục Y", numeric_cols, index=1)

x_idx = numeric_cols.index(col_x)
y_idx = numeric_cols.index(col_y)

fig2, ax2 = plt.subplots()

colors = plt.cm.viridis(np.linspace(0, 1, chosen_k))

for cluster in range(chosen_k):

    cluster_data = df[df['Cluster'] == cluster]
    cluster_scaled = df_scaled.loc[cluster_data.index]

    ax2.scatter(
        cluster_scaled.iloc[:, x_idx],
        cluster_scaled.iloc[:, y_idx],
        c=[colors[cluster]],
        s=120,
        label=f"Cụm {cluster}"
    )

centers = kmeans.cluster_centers_

ax2.scatter(
    centers[:, x_idx],
    centers[:, y_idx],
    marker='X',
    s=300,
    c='black',
    label='Tâm cụm'
)

for i, name in enumerate(df['Tên']):
    ax2.annotate(
        name,
        (df_scaled.iloc[i, x_idx], df_scaled.iloc[i, y_idx])
    )

ax2.set_xlabel(col_x)
ax2.set_ylabel(col_y)
ax2.set_title("Phân cụm K-Means")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# =========================
# 6. Thống kê
# =========================

st.subheader("📊 Trung bình đặc trưng theo cụm")

cluster_means = df.groupby('Cluster')[numeric_cols].mean().round(1)

st.dataframe(cluster_means, use_container_width=True)

st.subheader("🧠 Diễn giải kết quả phân cụm")

st.markdown("""
**Cụm 0 – Người bơi chuyên nghiệp**

- Số buổi bơi/tuần cao  
- Thời gian bơi dài  
- Tham gia nhiều lớp huấn luyện  

➡ Đây là nhóm hội viên có mức độ luyện tập cao và thường xuyên.

---

**Cụm 1 – Người mới**

- Số buổi bơi ít  
- Thời gian bơi ngắn  
- Ít hoặc không tham gia lớp học  

➡ Đây là nhóm hội viên mới bắt đầu hoặc bơi giải trí.

---

**Cụm 2 – Nhóm trung bình**

- Tần suất bơi và thời gian ở mức trung bình  
- Có tham gia một số lớp học  

➡ Đây là nhóm hội viên bơi khá thường xuyên nhưng chưa ở mức chuyên nghiệp.
""")
