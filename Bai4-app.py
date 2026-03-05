import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

st.set_page_config(page_title="Báo cáo phân cụm hội viên bể bơi", layout="wide")

st.title("📊 Báo cáo phân cụm hội viên bể bơi")

# 1. Dữ liệu từ file PDF
@st.cache_data
def load_data():
    data = {
        'Tên': ['An', 'Bình', 'Cường', 'Dung', 'Em', 'Phong', 'Giang', 'Hà',
                'Khánh', 'Linh', 'Minh', 'Ngọc', 'Phúc', 'Quỳnh', 'Sơn'],
        'Sốbuổi bơi/tuần': [2, 5, 7, 1, 4, 8, 3, 6, 9, 2, 5, 7, 3, 6, 10],
        'Thời gian TB (phút)': [30, 60, 75, 20, 45, 90, 35, 65, 85, 25, 55, 80, 40, 70, 95],
        'Sốlớp tham gia/tuần': [0, 2, 3, 0, 1, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5]
    }
    return pd.DataFrame(data)

df = load_data()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Dữ liệu gốc")
    st.dataframe(df, use_container_width=True)
with col2:
    st.subheader("📈 Gợi ý từ file")
    st.write("**k = 3** (Elbow method)")  # [file:1]
    st.write("**Cụm 0**: Chuyên nghiệp")  # [file:1]
    st.write("**Cụm 1**: Người mới")  # [file:1]
    st.write("**Cụm 2**: Trung bình")  # [file:1]

# 2. Chuẩn hóa
numeric_cols = ['Sốbuổi bơi/tuần', 'Thời gian TB (phút)', 'Sốlớp tham gia/tuần']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

st.subheader("🔄 Dữ liệu đã chuẩn hóa")
st.dataframe(df_scaled.head(10))

# 3. Elbow Method
st.subheader("📉 Elbow Method")

max_k = st.slider("k tối đa:", 5, 10, 8)
inertia_values = []
k_range = range(1, max_k + 1)

fig, ax = plt.subplots(figsize=(8, 5))
for k in k_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_tmp.fit(df_scaled)
    inertia_values.append(kmeans_tmp.inertia_)

ax.plot(k_range, inertia_values, marker='o', linewidth=2, markersize=8)
ax.set_title("Elbow Method - Điểm gãy tại k=3")
ax.set_xlabel("Số cụm (k)")
ax.set_ylabel("Inertia")
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# 4. K-Means
chosen_k = st.number_input("Chọn k:", 2, 8, 3)

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

st.subheader("🏷️ Kết quả phân cụm")
st.dataframe(df[['Tên', 'Cluster']], use_container_width=True)

# 5. Scatter plot - KHÔNG dùng seaborn
st.subheader("🎨 Biểu đồ phân cụm")

col_x = st.selectbox("Trục X:", numeric_cols, index=0)
col_y = st.selectbox("Trục Y:", numeric_cols, index=1)

x_idx = numeric_cols.index(col_x)
st.subheader("🎨 Biểu đồ phân cụm")

col_x = st.selectbox("Trục X:", numeric_cols, index=0)
col_y = st.selectbox("Trục Y:", numeric_cols, index=1)

x_idx = numeric_cols.index(col_x)
y_idx = numeric_cols.index(col_y)

fig2, ax2 = plt.subplots(figsize=(10, 7))

# Màu sắc cho từng cụm - FIX LỖI iloc
colors = plt.cm.viridis(np.linspace(0, 1, chosen_k))
for cluster in range(chosen_k):
    # DÙNG .loc thay vì .iloc với mask boolean
    cluster_data = df[df['Cluster'] == cluster]
    cluster_scaled = df_scaled.loc[cluster_data.index]
    
    ax2.scatter(
        cluster_scaled.iloc[:, x_idx], 
        cluster_scaled.iloc[:, y_idx],
        c=[colors[cluster]], 
        s=150, 
        alpha=0.8,
        edgecolors='black',
        linewidth=1,
        label=f'Cụm {cluster}'
    )

# Tâm cụm
centers = kmeans.cluster_centers_
ax2.scatter(
    centers[:, x_idx], 
    centers[:, y_idx],
    marker='X', 
    s=300, 
    c='black', 
    linewidth=3, 
    label='Tâm cụm'
)

# Ghi nhãn tên
for i, name in enumerate(df['Tên']):
    ax2.annotate(
        name, 
        (df_scaled.iloc[i, x_idx], df_scaled.iloc[i, y_idx]),
        xytext=(3, 3), 
        textcoords='offset points',
        fontsize=10, 
        ha='left'
    )

ax2.set_title(f"Phân cụm K-Means (k={chosen_k})")
ax2.set_xlabel(f'Scaled {col_x}')
ax2.set_ylabel(f'Scaled {col_y}')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
ax2.scatter(
    centers[:, x_idx], 
    centers[:, y_idx],
    marker='X', 
    s=300, 
    c='black', 
    linewidth=3, 
    label='Tâm cụm'
)

# Ghi nhãn tên
for i, name in enumerate(df['Tên']):
    ax2.annotate(
        name, 
        (df_scaled.iloc[i, x_idx], df_scaled.iloc[i, y_idx]),
        xytext=(3, 3), 
        textcoords='offset points',
        fontsize=10, 
        ha='left'
    )

ax2.set_title(f"Phân cụm K-Means (k={chosen_k})")
ax2.set_xlabel(f'Scaled {col_x}')
ax2.set_ylabel(f'Scaled {col_y}')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

# 6. Thống kê
st.subheader("📊 Thống kê trung bình theo cụm")
cluster_means = df.groupby('Cluster')[numeric_cols].mean().round(1)
st.dataframe(cluster_means, use_container_width=True)

st.success("✅ Báo cáo hoàn chỉnh! Dựa trên file PDF của bạn.")  # [file:1]
