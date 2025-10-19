import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import numpy as np
import os

from pull_prices import merged_df
from params import mcp, mdp, e, fee, nodes, products

# Create output directory for plots
output_dir = os.path.join(os.getcwd(), 'optimization_results')
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved to: {output_dir}\n")

# Read the data
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])

# Define sets and parameters
model = pyo.ConcreteModel()

model.t = pyo.Set(initialize=range(1, len(merged_df) + 1))
model.p = pyo.Set(initialize=products)
model.n = pyo.Set(initialize=nodes)

# ============================================================================
# CORRECTED PRICE DATA LOADING - Load for all nodes
# ============================================================================
model.price = pyo.Param(model.t, model.p, model.n, mutable=True)
for t in model.t:
    for p in model.p:
        for n in model.n:
            model.price[t, p, n] = merged_df.loc[t - 1, f"{p}"]

# ============================================================================
# DECISION VARIABLES
# ============================================================================
# Binary variable: 1 = buy mode, 0 = sell mode (mutually exclusive per hour)
model.is_buy = pyo.Var(model.t, model.n, within=pyo.Binary, initialize=0)

# Continuous buy/sell quantities
model.buy = pyo.Var(model.t, model.p, model.n, bounds=(0, None), initialize=0)
model.sell = pyo.Var(model.t, model.p, model.n, bounds=(0, None), initialize=0)

# Battery state of charge (energy stored)
model.soc = pyo.Var(model.t, bounds=(0, mcp), initialize=mcp)

# Net charge/discharge from battery (can be positive or negative)
model.net_energy_flow = pyo.Var(model.t, bounds=(-mdp, mcp), initialize=0)

# ============================================================================
# CONSTRAINTS
# ============================================================================

# 1. BATTERY STATE OF CHARGE DYNAMICS
# Energy stored changes based on net flow and efficiency
def battery_dynamics(model, t):
    if t == 1:
        # Initial state: fully charged
        return model.soc[t] == mcp + model.net_energy_flow[t]
    else:
        # Energy stored = previous stored + (charging * efficiency) - (discharging / efficiency)
        return model.soc[t] == model.soc[t-1] + model.net_energy_flow[t]

model.battery_dynamics_constr = pyo.Constraint(model.t, rule=battery_dynamics)

# 2. BATTERY BOUNDS
def soc_min(model, t):
    return model.soc[t] >= 0

def soc_max(model, t):
    return model.soc[t] <= mcp

model.soc_min_constr = pyo.Constraint(model.t, rule=soc_min)
model.soc_max_constr = pyo.Constraint(model.t, rule=soc_max)

# 3. NET ENERGY FLOW CALCULATION
# Energy bought must account for efficiency; energy sold gets efficiency benefit
def net_flow_calc(model, t, n):
    energy_bought = model.buy[t, "SP15", n]
    energy_sold = model.sell[t, "SP15", n]
    
    # When charging: store energy * efficiency
    # When discharging: withdraw energy / efficiency (lose efficiency)
    charge_contribution = energy_bought * e
    discharge_contribution = energy_sold / e
    
    # AS products don't consume real energy, they're virtual commitments
    # They affect battery dispatch constraints but not SOC directly
    as_charge = sum(model.buy[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
    as_discharge = sum(model.sell[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
    
    return model.net_energy_flow[t] == charge_contribution - discharge_contribution

model.net_flow_calc_constr = pyo.Constraint(model.t, model.n, rule=net_flow_calc)

# 4. MUTUAL EXCLUSIVITY: Cannot buy and sell simultaneously
# If is_buy[t] = 1, can only buy (sell must be 0)
# If is_buy[t] = 0, can only sell (buy must be 0)
def buy_sell_exclusivity_buy(model, t, p, n):
    return model.buy[t, p, n] <= (mcp if p == "SP15" else mcp) * model.is_buy[t, n]

def buy_sell_exclusivity_sell(model, t, p, n):
    return model.sell[t, p, n] <= (mdp if p == "SP15" else mdp) * (1 - model.is_buy[t, n])

model.buy_exclusivity = pyo.Constraint(model.t, model.p, model.n, rule=buy_sell_exclusivity_buy)
model.sell_exclusivity = pyo.Constraint(model.t, model.p, model.n, rule=buy_sell_exclusivity_sell)

# 5. CHARGE RATE LIMITS
# Cannot charge more than available capacity
def charge_rate_limit(model, t, n):
    available_capacity = (mcp - model.soc[t]) / e if t > 1 else (mcp - mcp) / e
    return model.buy[t, "SP15", n] <= available_capacity + mcp  # +mcp to avoid infeasibility at t=1

model.charge_rate_constr = pyo.Constraint(model.t, model.n, rule=charge_rate_limit)

# 6. DISCHARGE RATE LIMITS
# Cannot discharge more than available energy
def discharge_rate_limit(model, t, n):
    available_energy = model.soc[t] * e
    return model.sell[t, "SP15", n] <= available_energy

model.discharge_rate_constr = pyo.Constraint(model.t, model.n, rule=discharge_rate_limit)

# 7. ANCILLARY SERVICES CONSTRAINTS
# AS products require minimum commitment when bid
# Cannot exceed total battery capacity for AS
def as_total_limit(model, t, n):
    total_as_buy = sum(model.buy[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
    total_as_sell = sum(model.sell[t, p, n] for p in ["RegUp", "Spin", "RegDown", "NonSpin"])
    return total_as_buy + total_as_sell <= mcp

model.as_total_limit_constr = pyo.Constraint(model.t, model.n, rule=as_total_limit)

# ============================================================================
# OBJECTIVE FUNCTION: Maximize profit
# ============================================================================
def objective(model):
    profit = 0
    for t in model.t:
        for p in model.p:
            for n in model.n:
                # Revenue from selling
                sell_revenue = model.sell[t, p, n] * (model.price[t, p, n] - fee)
                
                # Cost from buying
                buy_cost = model.buy[t, p, n] * (model.price[t, p, n] + fee)
                
                profit += sell_revenue - buy_cost
    
    return profit

model.objective = pyo.Objective(rule=objective, sense=pyo.maximize)

# ============================================================================
# SOLVE THE MODEL
# ============================================================================
solverpath_exe = 'C://Users//groutgauss//anaconda3//pkgs//glpk-5.0-h8ffe710_0//Library//bin//glpsol.exe'
solver = pyo.SolverFactory('glpk', executable=solverpath_exe)

# Add solver options for faster convergence
solver_options = {
    'tmlim': 300,  # Time limit in seconds (5 minutes)
    'mipgap': 0.05,  # Stop when 5% gap is reached
}

results = solver.solve(model, tee=True, options=solver_options)

# ============================================================================
# EXTRACT AND VALIDATE RESULTS
# ============================================================================
if results.solver.status == pyo.SolverStatus.ok:
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION SUCCESSFUL")
    print(f"{'='*80}")
    print(f"Total Profit: ${model.objective():.2f}")
    print(f"{'='*80}\n")
    
    # Extract results into DataFrames for easier analysis
    results_data = []
    
    for t in model.t:
        for p in model.p:
            for n in model.n:
                results_data.append({
                    'time_step': t,
                    'datetime': merged_df.loc[t-1, 'datetime'],
                    'product': p,
                    'node': n,
                    'price': model.price[t, p, n].value,
                    'buy_qty': model.buy[t, p, n].value,
                    'sell_qty': model.sell[t, p, n].value,
                    'is_buy': model.is_buy[t, n].value if p == products[0] else None,
                })
    
    results_df = pd.DataFrame(results_data)
    
    # Battery state and net flow
    battery_results = []
    for t in model.t:
        battery_results.append({
            'time_step': t,
            'datetime': merged_df.loc[t-1, 'datetime'],
            'soc': model.soc[t].value,
            'net_flow': model.net_energy_flow[t].value,
        })
    
    battery_df = pd.DataFrame(battery_results)
    
    # Calculate hourly P&L by product
    pnl_data = []
    for t in model.t:
        hourly_pnl = 0
        for p in model.p:
            for n in model.n:
                sell_pnl = model.sell[t, p, n].value * (model.price[t, p, n].value - fee)
                buy_pnl = -model.buy[t, p, n].value * (model.price[t, p, n].value + fee)
                hourly_pnl += sell_pnl + buy_pnl
        
        pnl_data.append({
            'time_step': t,
            'datetime': merged_df.loc[t-1, 'datetime'],
            'hourly_pnl': hourly_pnl,
        })
    
    pnl_df = pd.DataFrame(pnl_data)
    pnl_df['cumulative_pnl'] = pnl_df['hourly_pnl'].cumsum()
    
    print("\nHourly P&L Summary (first 10 hours):")
    print(pnl_df.head(10).to_string(index=False))
    print(f"\nTotal P&L: ${pnl_df['cumulative_pnl'].iloc[-1]:.2f}")
    
else:
    print("Error: Solving failed.")
    print(f"Solver Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")

# ============================================================================
# PLOTTING AND VISUALIZATION
# ============================================================================

# Clean data
merged_df = merged_df.drop(columns=['RegDownMileage', 'RegUpMileage'], errors='ignore')

# 1. Energy and AS Prices
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# SP15 Energy Price
ax1.plot(merged_df['datetime'], merged_df['SP15'], label='SP15', color='dodgerblue', linewidth=2)
ax1.set_ylabel('Energy Price ($/MWh)', fontsize=12)
ax1.set_title('CAISO Energy LMP @ SP15', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# AS Prices
as_prices = ['RegUp', 'Spin', 'NonSpin', 'RegDown']
for price in as_prices:
    ax2.plot(merged_df['datetime'], merged_df[price], label=price, linewidth=2)

ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('AS Price ($/MWh)', fontsize=12)
ax2.set_title('CAISO Ancillary Service Prices', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_prices.png'), dpi=300, bbox_inches='tight')
print("Saved: 01_prices.png")
plt.close()

# 2. Battery State of Charge
plt.figure(figsize=(15, 6))
plt.plot(battery_df['datetime'], battery_df['soc'], linestyle='-', color='dodgerblue', linewidth=2)
plt.fill_between(battery_df['datetime'], 0, battery_df['soc'], alpha=0.3, color='dodgerblue')
plt.axhline(y=mcp, color='red', linestyle='--', label=f'Max Capacity ({mcp} MWh)', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title("Battery State of Charge Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=12)
plt.ylabel("SOC (MWh)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_battery_soc.png'), dpi=300, bbox_inches='tight')
print("Saved: 02_battery_soc.png")
plt.close()

# 3. Buy/Sell Dispatch by Product
fig, axs = plt.subplots(len(products), 1, figsize=(15, 4*len(products)))
if len(products) == 1:
    axs = [axs]

for idx, p in enumerate(products):
    product_results = results_df[results_df['product'] == p]
    buys = product_results['buy_qty'].values
    sells = -product_results['sell_qty'].values  # Negate for visualization
    
    axs[idx].bar(battery_df['datetime'], buys, label='Buy', color='indianred', alpha=0.7, width=0.02)
    axs[idx].bar(battery_df['datetime'], sells, label='Sell', color='dodgerblue', alpha=0.7, width=0.02)
    axs[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axs[idx].set_title(f'Product: {p}', fontsize=12, fontweight='bold')
    axs[idx].set_ylabel('Quantity (MWh)', fontsize=11)
    axs[idx].legend(fontsize=10)
    axs[idx].grid(True, alpha=0.3, axis='y')

axs[-1].set_xlabel('Time', fontsize=12)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, '03_buysell_dispatch.png'), dpi=300, bbox_inches='tight')
print("Saved: 03_buysell_dispatch.png")
plt.close()

# 4. Hourly and Cumulative Profit
fig, ax1 = plt.subplots(figsize=(15, 6))

ax1.bar(pnl_df['datetime'], pnl_df['hourly_pnl'], label='Hourly Profit', 
        color=['green' if x > 0 else 'red' for x in pnl_df['hourly_pnl']], alpha=0.7, width=0.02)
ax1.set_ylabel("Hourly Profit ($)", fontsize=12, color='black')
ax1.set_title("Hourly and Cumulative Profit", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()
ax2.plot(pnl_df['datetime'], pnl_df['cumulative_pnl'], label='Cumulative Profit', 
         color='darkblue', linewidth=3, marker='o', markersize=3)
ax2.set_ylabel("Cumulative Profit ($)", fontsize=12, color='darkblue')
ax2.tick_params(axis='y', labelcolor='darkblue')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_profit_analysis.png'), dpi=300, bbox_inches='tight')
print("Saved: 04_profit_analysis.png")
plt.close()

# 5. Net Energy Flow (Charge/Discharge)
fig, ax = plt.subplots(figsize=(15, 6))
colors = ['green' if x > 0 else 'red' for x in battery_df['net_flow']]
ax.bar(battery_df['datetime'], battery_df['net_flow'], color=colors, alpha=0.7, width=0.02, 
       label='Net Energy Flow')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_title("Net Energy Flow (Positive=Charging, Negative=Discharging)", fontsize=14, fontweight='bold')
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Energy Flow (MWh)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_net_energy_flow.png'), dpi=300, bbox_inches='tight')
print("Saved: 05_net_energy_flow.png")
plt.close()

print(f"\n{'='*80}")
print(f"All plots saved to: {output_dir}")
print(f"{'='*80}")