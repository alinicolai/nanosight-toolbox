

import os
import numpy as np
import pandas
from csv import reader


def read_experiment_summary_file(filepath):
 
    """
    read an Nanosight export ExperimentSummary.csv file
        
        parameters
        ----------
        filepath: path of the file
    
        returns
        ----------
        a pandas dataframe containing size distributions
        a pandas dataframe containing size attributes (mean, mode, etc.) and concentration attributes (total concentration)
        a pandas dataframe containing experiment infos (particles per frame, etc.)
        
    """

    if not os.path.exists(filepath):
        raise ValueError("File not found", filepath)

    """
    read the csv file to find where data start and end
    """

    # open file in read mode
    with open(filepath, 'r', encoding="ISO-8859-1") as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # iterate over each row in the csv using reader object
        rows_list = []
        for row in csv_reader:
            # rows_list contains each row of the csv
            rows_list.append(row)

    # transform each row in character string to help processing
    rows_list = np.array(["-".join(row) for row in rows_list]).astype(str)

    # find the row index marking the start of the size distribution data
    is_start = [True if row=='Graph Data-' else False for row in rows_list]
    start_index = np.where(is_start)[0][0]
        
    """
    extract size distributions data
    """

    # extract size distributions data as a pandas dataframe
    size_distributions = pandas.read_csv(filepath, skiprows=start_index+1, sep=",", 
                                         encoding = "ISO-8859-1", header=0)

    # drop unnamed column
    for col in [column for column in size_distributions.columns if "Unnamed:" in column]:
        size_distributions.drop(col, axis=1, inplace=True)

    # search stop index
    for i in range(len(size_distributions)):
        if str(size_distributions.iloc[i]["Bin centre (nm)"])=="nan" or str(size_distributions.iloc[i]["Bin centre (nm)"])=="Percentile":
            stop_index = i   
            break
        
    # slice the dataframe to end at stop index
    size_distributions = size_distributions.iloc[:stop_index]

    # convert dataframe to float as it contains only numerical information
    size_distributions = size_distributions.astype(float)
    
    n_videos = len([col for col in size_distributions.columns if 'Concentration (particles / ml)' in col])

    # rename the columns to make the names more explicit and concise."
    videos_cols = [col for col in size_distributions.columns if "Concentration (particles / ml)" in col] 
    new_videos_cols = ['Concentration Video '+str(k+1) for k in range(len(videos_cols))]
    size_distributions.rename(columns = {videos_cols[k]: new_videos_cols[k] for k in range(len(videos_cols))}, inplace=True)
    
    # drop nanosight average and standard errors calculations, we will recalculate everything
    # they are present only for non autosampler exports
    if 'Concentration average' in size_distributions.columns:
        size_distributions.drop(columns=['Concentration average', 'Standard Error'], axis=1, inplace=True)
        
    # columns are now : [Bin centre (nm), Concentration Video 1, ..., Concentration Video 5]

    """
    extract other results
    """
    
    # details_experiment = pandas.read_csv(filepath, sep=",", usecols=range(2), encoding = "ISO-8859-1")      

    # extract all available results        
    is_start_experiment_infos = [True if '[Results]' in row else False for row in rows_list]
    start_experiment_infos_index = np.where(is_start_experiment_infos)[0][0]
    experiment_infos = pandas.read_csv(filepath, sep=",", skiprows=start_experiment_infos_index, usecols=range(n_videos+1), encoding = 'ISO-8859-1')   
    experiment_infos.columns = ['key'] + ['Video '+str(k+1) for k in range(n_videos)]
    
    # extract size_data
    index_start_size_data = np.where(experiment_infos['key']=='[Size Data]')[0][0]
    index_end_size_data = np.where(experiment_infos['key']=='Graph Data')[0][0]
    size_concentration_attributes = experiment_infos[index_start_size_data+1:index_end_size_data]
    information_indexes = [True if key in ['Mean', 'Mode', 'SD', 'D50', 'D90'] else False for key in size_concentration_attributes['key']]

    size_concentration_attributes = size_concentration_attributes[information_indexes]
    size_concentration_attributes['key'] = size_concentration_attributes['key'].apply(lambda s: s + ' size')
    
    # # add total concentration
    # not necessary because we recalculate it later in nanosight_data_extraction.py
    # key_of_interest = ['Concentration (Particles / ml)']
    # key_of_interest_indexes = [True if key in key_of_interest else False for key in experiment_infos['key']]
    # size_concentration_attributes = pandas.concat([experiment_infos[key_of_interest_indexes], size_concentration_attributes])
    # # rename the column
    # size_concentration_attributes['key'] = size_concentration_attributes['key'].apply(lambda s: 
    #                                             s.replace('Concentration (Particles / ml)', 'Total concentration'))

    # extract metadata results
    key_of_interest = ['Particles per frame', 'Noise level']
    key_of_interest_indexes = [True if key in key_of_interest else False for key in experiment_infos['key']]
    experiment_infos = experiment_infos[key_of_interest_indexes]

    # keep only first row of particles frame and noise level, which contains the desired information
    particles_per_frame_rows = (experiment_infos['key']=='Particles per frame')
    indexes = np.where(particles_per_frame_rows)[0]
    experiment_infos.drop(experiment_infos.index[indexes[1]], axis=0, inplace=True)
    particles_per_frame_rows = (experiment_infos['key']=='Noise level')
    indexes = np.where(particles_per_frame_rows)[0]
    experiment_infos.drop(experiment_infos.index[indexes[1]], axis=0, inplace=True)
    
    # regenerate indexes for easier manipulation
    size_concentration_attributes.reset_index(inplace=True, drop=True)

    experiment_infos.reset_index(inplace=True, drop=True)

    return size_distributions, size_concentration_attributes, experiment_infos



