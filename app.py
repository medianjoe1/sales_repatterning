import streamlit as st
import pandas as pd
from repattern_model import sales_repattern_optimize
from max_sales_model import max_feasible_sales

st.set_page_config(layout="wide")

st.title("Sales Repatterning")
st.write("""
    This application allows users to interact with the sales repattern optimization.
    Test out different inputs and see how the recommended repatterning changes. 
    You can copy/paste values into the below input tables.
""")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Default Data Initialization ---
def init_sales_inventory_df():
    return pd.DataFrame(
        {m: [10000, 10000, 7500] for m in months},
        index=["Original Sales", "Wholesales", "Dealer Stock"]
    )

def init_constraints_df():
    return pd.DataFrame(
        {m: [15, 60] for m in months},
        index=["Min DoS", "Max DoS"]
    )

def init_operational_df():
    return pd.DataFrame(
        {m: [0, 24] for m in months},
        index=["Frozen Months (1=Frozen, 0=Avail to Repattern)", "Selling Days"]
    )

# --- User Inputs ---
st.header("Model Inputs")
sales_target = st.number_input("Total Annual Sales Target", value=150000, step=100)

with st.expander("📈 Monthly Sales & Inventory Inputs", expanded=True):
    sales_inventory_df = st.data_editor(
        init_sales_inventory_df(),
        use_container_width=True,
        num_rows="fixed"
    )

with st.expander("📊 Inventory Targets", expanded=False):
    constraints_df = st.data_editor(
        init_constraints_df(),
        use_container_width=True,
        num_rows="fixed"
    )

with st.expander("⚙️ Frozen Months & Selling Days", expanded=False):
    operational_df = st.data_editor(
        init_operational_df(),
        use_container_width=True,
        num_rows="fixed"
    )
    # alpha = st.number_input("Alpha (weight on spread)", value=1.0)
    # lambda_penalty = st.number_input("Lambda Penalty (on target deviation)", value=10.0)

# --- Extract Inputs from DataFrames ---
original_sales = sales_inventory_df.loc["Original Sales"].tolist()
wholesales = sales_inventory_df.loc["Wholesales"].tolist()
dealer_stock = sales_inventory_df.loc["Dealer Stock"].tolist()

min_dos_targets = constraints_df.loc["Min DoS"].tolist()
max_dos_targets = constraints_df.loc["Max DoS"].tolist()

frozen_months = [int(x) for x in operational_df.loc["Frozen Months (1=Frozen, 0=Avail to Repattern)"].tolist()]
selling_days = operational_df.loc["Selling Days"].tolist()

# --- Display Results ---
def display_results(result, original_sales, sales_target):
    # Summary
    summary_df = pd.DataFrame(
        {"Total Sales": 
         [max_sales['max_total_sales'], 
          sum(original_sales), 
          sales_target,  
          sum(result["final_sales"]), 
          result["target_deviation"]]
          },
        index=["Max Sales Push", "Original", "Target",  "Final Repattern", "Gap to Target"]
    )
    st.write("### 📊 Summary")
    st.dataframe(summary_df.style.format("{:,.0f}"), use_container_width=False)

    # Compute difference by month: Original - Final
    sales_diff = [
        orig - final
        for orig, final in zip(original_sales, result["final_sales"])
    ]

    # Monthly Breakdown
    results_table = pd.DataFrame(
        [
            original_sales,
            result.get("final_sales", [None]*12),
            sales_diff,
            result.get("inventory_levels", [None]*12),
            max_dos_targets,
            result.get("days_of_supply", [None]*12),
            min_dos_targets
        ],
        index=[
            "Original Sales",
            "Final Sales",
            "Δ Sales (Original - Final)",
            "Final Dealer Stock",
            "Max DoS Target",
            "Final Days of Supply",
            "Min DoS Target",
        ],
        columns=months
    )

    st.write("### 📈 Monthly Breakdown")
    st.dataframe(results_table.style.format("{:,.0f}"), use_container_width=True)


# --- Run Optimization ---
if st.button("Run Optimization"):
    max_sales = max_feasible_sales(
        original_sales = original_sales,
        min_dos_targets = min_dos_targets,
        max_dos_targets = max_dos_targets,
        wholesales = wholesales,
        dealer_stock = dealer_stock,
        frozen_months = frozen_months,
        selling_days = selling_days
    )

    result = sales_repattern_optimize(
        original_sales=original_sales,
        sales_target=sales_target,
        min_dos_targets=min_dos_targets,
        max_dos_targets=max_dos_targets,
        wholesales=wholesales,
        dealer_stock=dealer_stock,
        frozen_months=frozen_months,
        selling_days=selling_days,
        # alpha=alpha,
        # lambda_penalty=lambda_penalty,
    )
    display_results(result, original_sales, sales_target)
