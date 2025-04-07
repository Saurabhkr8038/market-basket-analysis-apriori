# 🛒 Market Basket Analysis using Apriori Algorithm

This project uses Association Rule Mining to discover patterns in customer purchases using the Apriori algorithm. A memory-efficient, interactive dashboard is built with Streamlit to explore itemsets, frequent rules, and support-confidence metrics.

## 🔍 Key Features

- Apriori Algorithm for market basket insights
- Association rules visualization (support, confidence, lift)
- Interactive Streamlit Dashboard
- Lightweight and optimized for limited memory environments
- Dataset from [Kaggle: Online Retail II Dataset](https://www.kaggle.com/datasets/mcnultyamy/online-retail-ii-uci)

## 🧰 Technologies Used

- Python
- Pandas
- mlxtend (for Apriori)
- Streamlit
- Matplotlib / Seaborn

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/market-basket-analysis-apriori.git
cd market-basket-analysis-apriori
pip install -r requirements.txt
streamlit run app.py
