import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'ess_data.csv'
df = pd.read_csv(file_path)

# Convert the 'date' column to datetime format if it's not already
df['date'] = pd.to_datetime(df['date'])
# 18~20 :X 06~ 18
# Filter the DataFrame for a specific date range
start_date = '2023-09-11'  # Replace with your start date
end_date = '2023-09-18'    # Replace with your end date
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
# 주말 08~09, 16~17, 23~24

# Calculate the Grid values
df_filtered['grid'] = df_filtered['load'] - df_filtered['pv']

# Plotting the graph for the filtered data
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['date'], df_filtered['load'], label='Load')
plt.plot(df_filtered['date'], df_filtered['pv'], label='PV')
plt.plot(df_filtered['date'], df_filtered['grid'], label='Grid')

# Setting the labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title(f'Load, PV, and Grid from {start_date} to {end_date}')
plt.legend()

# Format the date labels to 'MM-DD HH' and set the frequency of the ticks
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=6))  # Show every 6 hours

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
