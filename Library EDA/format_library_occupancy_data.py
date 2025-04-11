import pandas as pd
import numpy as np  # Import numpy for rounding up

"""
Looking at my calendar from 2023/24, 
Week 1 (Start of TB1 Lectures): 25th September 2023 - 1 October 2023
Week 12 (End of TB1 Lectures): 11th December 2023 - 17th December 2023

TEACHING BLOCK 1

TB1 Teaching period : 25th Sep 2023 - 17th Dec 2023
TB1 Reading Week : 30th Oct 2023 - 5th Nov 2023
Winter Break : 18th Dec 2023 - 5th Jan 2024
TB1 Assessment Period : 8th Jan 2024 - 19th Jan 2024 


TEACHING BLOCK 2

TB2 Teaching period : 22nd Jan 2024 - 5th May 2024
TB2 Reading Week : 26th Feb 2024 - 3rd Mar 2024
Week 21 (Last week before Spring Break): 18th Mar 2024 - 24th Mar 2024
Spring Break : 25th Mar 2024 - 12th Apr 2024
Revision Week : 6th May 2024 - 12th May 2024
TB2 Assessment Period : 13th May 2024 - 31st May 2024






"""


# Load the Excel file
df_raw = pd.read_excel(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\Library occupancy data AY 23-24.xlsx")  # Change the filename accordingly




# # Define how to combine duplicate values
# def merge_values(series):
#     return ' '.join(series.dropna().astype(str).unique())  # Join unique non-null values

# Group by 'date' and 'time' while merging other column values
# df_grouped = df_raw.groupby(['Date', 'Time range'], as_index=False).agg(merge_values)

# df_grouped = df_raw.groupby(['Date', 'Time range'], as_index=False).agg('mean')


# Identify non-numeric and numeric columns dynamically
datetime_cols = ['Date']  # Specify datetime column
non_numeric_cols = ['Time range', 'W/C', 'Day']  # Other non-numeric columns
numeric_cols = df_raw.select_dtypes(include=['number']).columns  # Automatically detect numeric columns


# Define aggregation functions
agg_funcs = {col: 'first' for col in datetime_cols}  # Keep first occurrence of datetime
agg_funcs.update({col: lambda x: ', '.join(map(str, x.dropna().unique())) for col in non_numeric_cols})  # Merge text columns
# agg_funcs.update({col: lambda x: np.ceil(x.mean()) for col in numeric_cols})  # Take mean and round up
agg_funcs.update({col: lambda x: x.max() for col in numeric_cols}) # Take largest value

# Perform groupby with custom aggregation
df_grouped = df_raw.groupby(['Date', 'Time range'], as_index=False).agg(agg_funcs)

df_grouped_sorted = df_grouped.sort_values(by = 'ASSL', ascending=False)

# df_grouped = pd.read_csv(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\cleaned_library_occupancy_data_23_24.csv")


# print(df_grouped[5,6])

# print(df_grouped.dtypes)

# Save the cleaned data
df_grouped.to_csv("cleaned_library_occupancy_data_23_24.csv", index=False)

print("Done!")