import pandas as pd
import os   


instance_list_filename = 'instances_list.xlsx' 
dfi = pd.read_excel(instance_list_filename, index_col='name')

# Create a new DataFrame to store the results   
df_run_gap_EEV = pd.DataFrame(columns=['name', 'runTimeEEV', 'gapEEV'])
# Loop through each instance in the list
for instance_name in dfi.index:
    solved = dfi.loc[instance_name, 'solved']
    if not solved:
        df_run_gap_EEV.loc[len(df_run_gap_EEV)] = [instance_name,-1, -1]
        continue
    path = os.path.join('results', instance_name + '_mean_res.xlsx')
    print(f"Reading {path}")
    df = pd.read_excel(path, index_col='param')
    runTime = df.loc['runtime', 'value']
    gap = df.loc['objValue', 'value']
    df_run_gap_EEV.loc[len(df_run_gap_EEV)] = [instance_name,runTime, gap]
# Save the results to an Excel file
minimum_runTime = df_run_gap_EEV['runTimeEEV'].min()
maximum_runTime = df_run_gap_EEV['runTimeEEV'].max()
minimum_gap = df_run_gap_EEV['gapEEV'].min()
maximum_gap = df_run_gap_EEV['gapEEV'].max()
print(f"Minimum runTime: {minimum_runTime}")
print(f"Maximum runTime: {maximum_runTime}")
print(f"Minimum gap: {minimum_gap}")
print(f"Maximum gap: {maximum_gap}")
output_filename = 'runTimeEEV.xlsx'
df_run_gap_EEV.to_excel(output_filename, index=False)
print(f"Results saved to {output_filename}")