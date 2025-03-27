import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as mtick


def format_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Removes leading/trailing spaces
    df.drop(columns = ["Veterinary Sciences Library"], inplace= True)


    col_list = list(df.columns)
    x, y = col_list.index("Grace Reeves Study Centre"), col_list.index("Hawthorns & Brambles")
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    

    # Rename columns so libraries have the same names as in library_locations.csv
    library_data = pd.read_csv(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\library_locations.csv")
    library_names = library_data['LibraryName '].tolist()
    df.columns = list(df.columns[:4]) + library_names

    df_values = df.loc[:, 'Arts and Social Sciences':]
    df_values = df_values.apply(pd.to_numeric, errors='coerce')


    capacity = library_data["Capacity"].astype(int).to_numpy()
    df_values_percentage = df_values.div(capacity,axis = 1) * 100

    df_percentage_formatted = pd.concat([df.iloc[:,:4],df_values_percentage],axis = 1)

    return df_percentage_formatted


def week_formatting(df):

    # Remove '00:00:00' from all elements in 'W/C' col
    df['W/C'] = df['W/C'].to_frame().applymap(lambda x: str(x).replace(' 00:00:00', '').strip() if isinstance(x, str) else x)
    weeks = df['W/C'].unique()

    df.to_csv("week_formatted_percentage_library_occupancy_23_24.csv", index = False)

    return weeks


def weekly_percentage_full_all_libraries(df, upper_threshold = None, lower_threshold = None):
    
    weeks = df['W/C'].unique()

    weekly_percentage_df = pd.DataFrame(weeks, columns = ['W/C'])

    # Ensure the first column is in datetime format
    weekly_percentage_df['W/C'] = pd.to_datetime(weekly_percentage_df['W/C'])    
    weekly_percentage_df['W/C'] = weekly_percentage_df['W/C'].dt.strftime('%d-%m-%Y')

    if upper_threshold != None:
        for threshold in upper_threshold:
            weekly_percentage_df[f'{threshold}% full'] = np.nan

            for week in enumerate(weeks):
                
                current_week_data = df[df['W/C'] == week[1]]
                current_week_occupancy_values = current_week_data.loc[:,'Arts and Social Sciences':]

                # Flatten all values into a single Series and drop NaNs
                all_values = current_week_occupancy_values.values.flatten()
                all_values = pd.Series(all_values).dropna()

                
                # Compute proportion of values ≥ upper_threshold across all libraries
                overall_proportion = (all_values >= threshold).sum() / all_values.count()
                weekly_percentage_df.at[week[0],f'{threshold}% full'] = overall_proportion * 100

                # elif lower_threshold != None:
                #     # Compute proportion of values ≤ lower_threshold across all libraries
                #     overall_proportion = (all_values <= threshold).sum() / all_values.count()
                #     weekly_percentage_df.at[week[0],f'{threshold}% full'] = overall_proportion * 100

    
    # Plot proportion of time library occupancy exceeds threshold per academic week
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    legend_thresholds = weekly_percentage_df.columns[1:]
    print(weekly_percentage_df.dtypes)

    test = weekly_percentage_df.at[16,'W/C']

    # Plot each threshold on one figure
    for threshold in legend_thresholds:
        ax.plot(weekly_percentage_df['W/C'], weekly_percentage_df[threshold], marker='o', linestyle='-',label = threshold)
    
    

    ax.axvline(x=weekly_percentage_df.at[16,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:purple', label='Assessment period')
    ax.axvline(x=weekly_percentage_df.at[18,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:purple')    

    ax.axvline(x=weekly_percentage_df.at[34,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:purple')
    ax.axvline(x=weekly_percentage_df.at[37,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:purple')   

    ax.axvline(x=weekly_percentage_df.at[6,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:gray', label='Reading week')
    ax.axvline(x=weekly_percentage_df.at[7,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:gray')

    ax.axvline(x=weekly_percentage_df.at[23,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:gray')
    ax.axvline(x=weekly_percentage_df.at[24,'W/C'], linestyle='--', linewidth = 1.5, color = 'tab:gray')   

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim([0,100])
    ax.legend(title = 'Threshold occupancy')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)  

    # Labels and title
    plt.xlabel("Week",fontsize = 12)
    plt.ylabel("Proportion of time libraries exceed threshold occupancy", fontsize = 12)
    # # plt.title("Proportion of time library occupancy exceeds threshold per academic week")

    plt.tight_layout()
    plt.grid()
    plt.show()


    return weekly_percentage_df


def individuaL_weekly_percentage_full(df, upper_threshold = None, lower_threshold = None):
    # ASS_library_data = df.loc[:,:'Arts and Social Sciences']
    
    # weeks = ASS_library_data['W/C'].unique()
    weeks = df['W/C'].unique()

    df.loc[:].columns

    # Keys : Weeks of the academic year 
    # Values : percent of the time that the library occupancy is above upper threshold
    weekly_percentage_dict = {}

    for week in weeks:
        
        current_week_data = df[df['W/C'] == week]
        current_week_occupancy_values = current_week_data.loc[:,'Arts and Social Sciences':]

        # Flatten all values into a single Series and drop NaNs
        all_values = current_week_occupancy_values.values.flatten()
        all_values = pd.Series(all_values).dropna()

        # Compute proportion of values ≥ upper_threshold across all libraries
        overall_proportion = (all_values >= upper_threshold).sum() / all_values.count()
        weekly_percentage_dict[week] = overall_proportion * 100

    # Function to calculate the proportion of values ≥ 95
    def proportion_ge_95(col):
        return (col >= 95).sum() / col.notna().sum()

    # Apply function to each numerical column
    proportions = df.apply(proportion_ge_95)

    # for week in weeks:
    #     current_week_data = ASS_library_data[ASS_library_data['W/C'] == week]

    #     # Select the column (replace 'Column1' with your actual column name)
    #     col = current_week_data['Arts and Social Sciences']

    #     # Calculate the proportion of non-missing values that are ≥ 95
    #     proportion = (col >= upper_threshold).sum() / col.notna().sum()
    #     weekly_percentage_dict[week] = proportion * 100


    return weekly_percentage_dict


if __name__ == "__main__":

    path = r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\cleaned_library_occupancy_data_23_24.csv"
    
    df_percentage_formatted = format_data(path)

    df_percentage_formatted.to_csv("percentage_library_occupancy_23_24.csv", index=False)

    # df_percentage_formatted = pd.read_csv(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\percentage_library_occupancy_23_24.csv")    

    # weeks = week_formatting(df_percentage_formatted)

    # df = pd.read_csv(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\week_formatted_percentage_library_occupancy_23_24.csv")
    df = pd.read_csv(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase C\MDM3-phaseC\percentage_library_occupancy_23_24.csv")

    weekly_occupancy = weekly_percentage_full_all_libraries(df, upper_threshold = [25,50,75,95], lower_threshold = [10,25])

    print("DONE!")