import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
# --- PAGE CONFIG ---
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# -------------------- PAGE SETUP WITH BACKGROUND --------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e"); /* Stylish ecommerce image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);  /* Adds semi-transparent overlay for readability */
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🛒 Market Basket Analysis Dashboard")
# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Saurav Kumar\python files\UPWORK PROJECTS\Product recommendation system\archive\online_retail_II.csv", encoding='ISO-8859-1')
    df.dropna(subset=["Customer ID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_data()
st.subheader("Data Preview")
st.dataframe(df.head(10))

# --- FILTER FOR A COUNTRY & SAMPLE FOR MEMORY ---
country = st.selectbox("Select Country", df['Country'].unique())
df_country = df[df['Country'] == country].copy()
df_country = df_country.sort_values('InvoiceDate').tail(5000)  # Keep recent 5k entries

# --- TOP PRODUCTS VISUALIZATION ---
st.subheader("📈 Top Purchased Products")
top_items = df_country['Description'].value_counts().nlargest(10)
fig, ax = plt.subplots()
sns.barplot(x=top_items.values, y=top_items.index, palette='Blues_r', ax=ax)
ax.set_title("Top 10 Products by Frequency")
ax.set_xlabel("Frequency")
st.pyplot(fig)


# --- CREATE BASKET ---
basket = df_country.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# --- APRIORI ---
min_support = st.slider("Select Minimum Support", min_value=0.005, max_value=0.05, step=0.005, value=0.01)
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

# --- ASSOCIATION RULES ---
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

st.subheader("📦 Frequent Itemsets")
st.dataframe(frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True))

st.subheader("🔗 Association Rules")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False).reset_index(drop=True))

