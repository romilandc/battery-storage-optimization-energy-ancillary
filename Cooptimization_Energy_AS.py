import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

#from pyomo import *
from pull_prices import merged_df

# Battery parameters
mcp = 10  # Max Charge Power (MWh)
mdp = 10  # Max Discharge Power (MWh)
e = 0.80  # Round trip efficiency
fee = 1  # Trade fees on both Buy and Sell trades ($/MWh)

# Read the data (replace with your actual data loading logic)
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])

# Define sets and parameters
model = pyo.ConcreteModel()

T = len(merged_df)  # Number of time steps
products = ["SP15", "RegUp", "Spin", "RegDown", "NonSpin"]
nodes = ["TH_SP15_GEN-APND"]  # Replace with actual node list

model.t = pyo.Set(initialize=range(1, T + 1))
model.p = pyo.Set(initialize=products)
model.n = pyo.Set(initialize=nodes)

# Price data (assuming prices are in separate columns)
model.price = pyo.Param(model.t, model.p, model.n, mutable=True)
for t in model.t:
    for p in model.p:
        model.price[t, p, nodes[0]] = merged_df.loc[t - 1, f"{p}"]


# Variables
model.buy = pyo.Var(model.t, model.p, model.n, bounds=(0, mcp), initialize=0)
model.sell = pyo.Var(model.t, model.p, model.n, bounds=(0, mdp), initialize=0)
model.C = pyo.Var(model.t, bounds=(0, mcp), initialize=mcp)  # Battery state of charge

# Constraints

def storage_state(model, t):
    # Set initial state to full charge
    if t == 1:
        return model.C[t] == mcp
    else:
        return model.C[t] == (model.C[t - 1] + sum(model.buy[t, p, n] * e for p in model.p for n in model.n) -
                             sum(model.sell[t, p, n] / e for p in model.p for n in model.n))

model.storage_state_constr = pyo.Constraint(model.t, rule=storage_state)

# Limit on charging and discharging based on product type
def product_limit(model, t, p, n):
    if p == "Energy":
        return model.buy[t, p, n] <= (mcp - model.C[t]) / e
    else:  # RegUp, Spin, RegDown, NonSpin
        return model.buy[t, p, n] <= mcp

    # Adjust the constraint for discharge limits based on product type
    # (e.g., RegDown might have a lower discharge limit)

model.product_limit_constr = pyo.Constraint(model.t, model.p, model.n, rule=product_limit)

# Battery cannot discharge below zero
def min_storage(model, t):
    return model.C[t] >= 0

model.min_storage_constr = pyo.Constraint(model.t, rule=min_storage)

# Battery cannot store more than maximum capacity
def max_storage(model, t):
    return model.C[t] <= mcp

model.max_storage_constr = pyo.Constraint(model.t, rule=max_storage)

# Limit one buy or sell order per product per node per time slice
def single_order(model, t, p, n):
    return model.buy[t, p, n] + model.sell[t, p, n] <= 1

model.single_order_constr = pyo.Constraint(model.t, model.p, model.n, rule=single_order)

# OBJECTIVE DEFINITION
def objective(model):
    profit = sum(
        (model.sell[t, p, n] * (model.price[t, p, n] - fee)) -
        (model.buy[t, p, n] * (model.price[t, p, n] + fee))
        for t in model.t
        for p in model.p
        for n in model.n
    )
    return profit

model.objective = pyo.Objective(rule=objective, sense=pyo.maximize)

# Solve the model
solverpath_exe='C://Users//groutgauss//anaconda3//pkgs//glpk-5.0-h8ffe710_0//Library//bin//glpsol.exe'
solver = pyo.SolverFactory('glpk', executable=solverpath_exe)
results = solver.solve(model, tee=False)

# Optional: Extract and analyze results
if results.solver.status == pyo.SolverStatus.ok:
    # Print total profit
    print(f"Total profit: ${model.objective.expr()}")

    # Access buy/sell decisions and battery state of charge for further analysis
    buy_decisions = {}
    sell_decisions = {}
    battery_state = {}
    for t in model.t:
        buy_decisions[t] = {}
        sell_decisions[t] = {}
        battery_state[t] = model.C[t].value

    # Process and analyze results further (e.g., calculate profit by product, visualize decisions)
else:
    print("Error: Solving failed.")
    
    
##Plots and Analysis

# Extract results
buy_decisions = {}
sell_decisions = {}
battery_state = {}
profit_by_hour = []

for t in model.t:
    buy_decisions[t] = {}
    sell_decisions[t] = {}
    for p in model.p:
        for n in model.n:
            buy_decisions[t][p] = model.buy[t, p, n].value
            sell_decisions[t][p] = model.sell[t, p, n].value
    battery_state[t] = model.C[t].value

    # Calculate hourly profit
    hourly_profit = sum(
        (sell_decisions[t][p] * (model.price[t, p, nodes[0]].value - fee)) -
        (buy_decisions[t][p] * (model.price[t, p, nodes[0]].value + fee))
        for p in model.p
    )
    profit_by_hour.append(hourly_profit)

# Calculate cumulative profit
cumulative_profit = [sum(profit_by_hour[:i+1]) for i in range(len(profit_by_hour))]

# Plot battery state of charge over time
plt.figure(figsize=(15, 6))
plt.plot(merged_df['datetime'], battery_state.values(), linestyle='-', color='dodgerblue')
plt.title("Battery State of Charge Over Time")
plt.xlabel("Datetime")
plt.ylabel("Charge Level (MWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Buy/Sell decisions for each product
fig, axs = plt.subplots(nrows=len(model.p), ncols=1, figsize=(30, 6*len(model.p)), sharex=False)
fig.suptitle("Buy/Sell Decisions for Each Product")

for i, p in enumerate(model.p):
    buys = [buy_decisions[t][p] for t in model.t]
    sells = [-s for s in [sell_decisions[t][p] for t in model.t]]  # Negate sell quantities for plotting
    axs[i].plot(merged_df['datetime'], buys, label="Buy", color="indianred")
    axs[i].plot(merged_df['datetime'], sells, label="Sell", color="dodgerblue")
    axs[i].set_title(f"Product: {p}")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Quantity (MWh)")
    axs[i].legend()

plt.tight_layout()
plt.xticks(rotation=45)  
plt.show()

# Plot hourly and cumulative profit
fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(datetimes, profit_by_hour, label="Hourly Profit", color="dodgerblue")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Profit ($)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_title("Hourly and Cumulative Profit")
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

ax2 = ax1.twinx()
ax2.plot(datetimes, cumulative_profit, label="Cumulative Profit", color="indianred")
ax2.set_ylabel("Cumulative Profit ($)", color="black")
ax2.tick_params(axis="y", labelcolor="black")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.show()