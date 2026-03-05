import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Báo cáo phân cụm hội viên bể bơi", layout="wide")

st.title("Báo cáo phân cụm hội viên bể bơi")

# 1. Tạo DataFrame từ dữ liệu trong đề bài
data = {
    'Tên': ['An', 'Bình', 'Cường', 'Dung', 'Em', 'Phong', 'Giang', 'Hà',
            'Khánh', 'Linh', 'Minh', 'Ngọc', 'Phúc', 'Quỳnh', 'Sơn'],
    'Sốbuổi bơi/tuần': [2, 5, 7, 1, 4, 8, 3, 6, 9, 2, 5, 7, 3, 6, 10],
    'Thời gian TB (phút)': [30, 60, 75, 20, 45, 90, 35, 65, 85, 25, 55, 80, 40, 70, 95],
    'Sốlớp tham gia/tuần': [0, 2, 3, 0, 1, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5]
}

df = pd.DataFrame(data)

st.subheader("1. Dữ liệu gốc")
st.dataframe(df)

# 2. Chuẩn hóa dữ liệu
numeric_cols = ['Sốbuổi bơi/tuần', 'Thời gian TB (phút)', 'Sốlớp tham gia/tuần']
X = df[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

st.subheader("2. Dữ liệu đã chuẩn hóa (StandardScaler)")
st.dataframe(df_scaled)

# 3. Elbow Method
st.subheader("3. Elbow Method - Chọn số cụm k")

max_k = st.slider("Chọn k tối đa để vẽ Elbow", min_value=5, max_value=10, value=8, step=1)

inertia_values = []
k_range = range(1, max_k + 1)

for k in k_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_tmp.fit(df_scaled)
    inertia_values.append(kmeans_tmp.inertia_)

fig_elbow, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_range), inertia_values, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Số cụm (k)")
ax.set_ylabel("Inertia")
ax.set_xticks(list(k_range))
ax.grid(True)

st.pyplot(fig_elbow)

st.write("Gợi ý trong file là chọn **k = 3** dựa trên điểm gãy của đường cong Elbow.")  # [file:1]

# 4. Chọn k và chạy K-Means
st.subheader("4. Phân cụm K-Means")

chosen_k = st.number_input("Chọn số cụm k", min_value=2, max_value=8, value=3, step=1)

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
kmeans.fit(df_scaled)

df['Cluster'] = kmeans.labels_

st.write("Bảng kết quả: Tên hội viên và cụm được phân vào")
st.dataframe(df[['Tên', 'Cluster']])

# 5. Scatter plot trực quan hóa
st.subheader("5. Biểu đồ scatter các cụm")

# Chọn hai đặc trưng để vẽ
col_x = st.selectbox("Chọn trục X", numeric_cols, index=0)
col_y = st.selectbox("Chọn trục Y", numeric_cols, index=1)

feature_x_idx = numeric_cols.index(col_x)
feature_y_idx = numeric_cols.index(col_y)

fig_scatter, ax2 = plt.subplots(figsize=(7, 5))

sns.scatterplot(
    x=df_scaled.iloc[:, feature_x_idx],
    y=df_scaled.iloc[:, feature_y_idx],
    hue=df['Cluster'],
    palette='viridis',
    s=100,
    alpha=0.8,
    ax=ax2
)

cluster_centers = kmeans.cluster_centers_
ax2.scatter(
    cluster_centers[:, feature_x_idx],
    cluster_centers[:, feature_y_idx],
    marker='X',
    s=200,
    color='black',
    edgecolor='black',
    linewidth=2,
    label='Tâm cụm'
)

# Ghi nhãn tên hội viên
for i, txt in enumerate(df['Tên']):
    ax2.annotate(
        txt,
        (df_scaled.iloc[i, feature_x_idx], df_scaled.iloc[i, feature_y_idx]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=8
    )

ax2.set_title(f"K-Means clustering (k={chosen_k})")
ax2.set_xlabel(f"Scaled {col_x}")
ax2.set_ylabel(f"Scaled {col_y}")
ax2.grid(True)
ax2.legend()

st.pyplot(fig_scatter)

# 6. Thống kê trung bình theo cụm
st.subheader("6. Thống kê cụm (giá trị trung bình các đặc trưng)")

cluster_means = df.groupby('Cluster')[numeric_cols].mean()
st.dataframe(cluster_means)

st.write("Bạn có thể dựa vào bảng trên để mô tả và đặt tên cho từng cụm như trong file PDF.")  # [file:1]

