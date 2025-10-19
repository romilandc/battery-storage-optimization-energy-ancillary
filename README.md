## Summary
A co-optimization battery storage model between Energy and Ancillary Service (AS) products [RegUp, Spin, RegDown, NonSpin] that finds a trading strategy to maximize revenue by Buying and Selling each hour.  For optimization we use Pyomo for model setup (see below constraints) and GLPK for solver.  You also have option to switch solvers to, e.g. gurobi or CBC.

We pull historical Energy and AS prices using the Gridstatus API (https://github.com/kmax12/gridstatus).  Append forecasted prices to the merged_df dataframe for a forecasted optimization.  Constraints described in the Cooptimization_Energy_As.py script are specified by the below assumptions:

## Battery Assumptions

- Maximum total charge level: 10 MWh
- Initial charge level: Fully charged
- Instantaneous charge/discharge
- Efficiency factor: 0.80 for both charge and discharge
- No simultaneous charging and discharging
- Battery cannot discharge more energy than available
- Battery cannot store more energy than maximum capacity
- No simultaneous charging and discharging

## Trading Assumptions

- Trading fees: $1 per MWh for both buy and sell transactions
- Buy/sell orders must be submitted one hour prior to execution
- Only one Buy or Sell order per time interval
- Cannot partcipate in multiple products at same time

## How to use
- create virtual environment using conda or .venv
- use git to clone repository `git clone https://github.com/romilan24/battery-storage-optimization-energy-ancillary`
- type `pip install -r /path/to/requirements.txt` in cmd prompt
- update path to local path where data is located
- update path to your solver `line 106` on Cooptimization_Energy_AS.py
- run script

## Example usecase
In our example, params are initizlied with: 

location = 'TH_SP15_GEN-APND'
start_date = 'Jan-01-2024'
end_date = 'Mar-01-2024' 

so just two months of prices.  Note that CAISO also has AS prices called 'RegDownMileage' and 'RegUpMileage' which is the cost for cycling the unit but we're not considering this in our example.

Running our script we see that "Total profit: $102,349.83" with the following plots:

## Energy Prices
![Image1](https://github.com/romilan24/battery-storage-optimization-energy-ancillary/blob/main/img/energy.png)

## AS Prices
![Image2](https://github.com/romilan24/battery-storage-optimization-energy-ancillary/blob/main/img/as.png)

## Battery State of Charge (SOC) which is the variation in battery level as we Buy & Sell
![Image3](https://github.com/romilan24/energy-ancillary-optimization/blob/main/img/batter_soc.png)

## Buy and Sell decisions for each Product
![Image4](https://github.com/romilan24/energy-ancillary-optimization/blob/main/img/Buy_Sell_per_Product.png)

## Hourly and Cumulative Profit
![Image5](https://github.com/romilan24/energy-ancillary-optimization/blob/main/img/hourly_Cum_Profit.png)
