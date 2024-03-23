nodes = ["TH_SP15_GEN-APND"] #add more nodes via ["TH_SP15_GEN-APND", "ANOTHER_NODE_ID"] etc
products = ["SP15", "RegUp", "Spin", "RegDown", "NonSpin"]
start_date = "Jan 1, 2024"
end_date = "Mar 1, 2024"
mcp = 10  # Max Charge Power (MWh)
mdp = 10  # Max Discharge Power (MWh)
e = 0.80  # Round trip efficiency
fee = 1  # Trade fees on both Buy and Sell trades ($/MWh)