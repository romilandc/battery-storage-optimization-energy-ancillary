import gridstatus
import pandas as pd
import matplotlib.pyplot as plt

from params import nodes, start_date, end_date

# Initialize
iso = gridstatus.CAISO()
lmp_final = pd.DataFrame()
as_final = pd.DataFrame()

def get_lmp_data(start, end, nodes):
    """
    Function to retrieve LMP data from CAISO.
    :param start: Start date (pandas Timestamp)
    :param end: End date (pandas Timestamp)
    :param nodes: List of location strings
    :return: DataFrame with LMP data
    """
    
    lmp = iso.get_lmp(start=start, end=end, market="DAY_AHEAD_HOURLY", locations=nodes, sleep=3)
    
    lmp_final = lmp.copy()
    lmp_final['Time'] = pd.to_datetime(lmp_final['Time'], utc=True)
    lmp_final['datetime'] = lmp_final['Time'].dt.tz_convert(None)
    lmp_final.drop(columns=['Energy', 'Congestion', 'Loss', 'Location', 'Time', 'Interval Start', 'Interval End', 'Market', 'Location Type'], inplace=True)
    lmp_final.rename(columns={'LMP': 'SP15'}, inplace=True)
    return lmp_final

def get_as_prices_data(start_date, end_date):
    """
    Function to retrieve AS prices data from CAISO.
    :param start_date: Start date (string)
    :param end_date: End date (string)
    :return: DataFrame with AS prices data
    """
    
    as_prices = iso.get_as_prices(date=start_date, end=end_date)

    as_final = as_prices.copy()
    as_final['Time'] = pd.to_datetime(as_final['Time'], utc=True)
    as_final['datetime'] = as_final['Time'].dt.tz_convert(None)
    as_final.drop(columns=['Time', 'Interval Start', 'Interval End'], inplace=True)
    as_final = as_final.groupby(['datetime']).agg({
        'Non-Spinning Reserves': 'sum',
        'Regulation Down': 'sum',
        'Regulation Mileage Down': 'sum',
        'Regulation Mileage Up': 'sum',
        'Regulation Up': 'sum',
        'Spinning Reserves': 'sum'
    }).reset_index()
    as_final.rename(columns={
        'Non-Spinning Reserves': 'NonSpin',
        'Regulation Down': 'RegDown',
        'Regulation Mileage Down': 'RegDownMileage',
        'Regulation Mileage Up': 'RegUpMileage',
        'Regulation Up': 'RegUp',
        'Spinning Reserves': 'Spin'
    }, inplace=True)
    return as_final

def get_merged_data(start_date, end_date, nodes):
    """
    Function to get merged LMP and AS prices data.
    :param start: Start date (pandas Timestamp) for LMP data
    :param end: End date (pandas Timestamp) for LMP data
    :param nodes: List of location strings for LMP data
    :param start_date: Start date (string) for AS prices data
    :param end_date: End date (string) for AS prices data
    :return: DataFrame with merged LMP and AS prices data
    """
    lmp_final = get_lmp_data(start_date, end_date, nodes)
    as_final = get_as_prices_data(start_date, end_date)

    merged_df = pd.merge(lmp_final, as_final, on='datetime', how='inner')

    # Reset the 'Hour' component of the 'datetime' column to start from 1 and reset at 24
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
    merged_df['datetime'] = merged_df['datetime'].dt.tz_localize('UTC')  # Convert to UTC first
    merged_df['datetime'] = merged_df['datetime'].dt.tz_convert('US/Pacific')  # Convert to Pacific time
    merged_df['datetime'] = merged_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Strip UTC component
    merged_df = merged_df.sort_values(by='datetime', inplace=False)  # Sort the dataframe by the updated 'datetime' column

    return merged_df


merged_df = get_merged_data(start_date, end_date, nodes)