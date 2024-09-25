
from pathlib import Path
import numpy as np
import pandas
import itertools


from data_extraction_module.nanosight_export_files_listing import list_nanosight_files_in_directory
from data_extraction_module.nanosight_export_files_reading import read_experiment_summary_file





def extract_nanosight_data_from_directory(directory_path, dilution_prefix, replicate_prefix):

    """
    extract all Nanosight data from a directory
        
        parameters
        ----------
        directory_path: path of the directory
        dilution_prefix: dilution prefix to consider when reading file names
        replicate_prefix: replicate prefix to consider when reading file names
    
        returns
        ----------
        a dictionary containing: 
        files_infos: a pandas dataframe containing files information (replicates if any, samples, dilutions)
        samples_filenames: a dictionary where the keys are sample names and the values are the corresponding list of replicates filenames
        size_distributions: a pandas dataframe containing size_distributions for all samples
        size_concentration_attributes: a pandas dataframe containing all size and concentration attributes for all samples
        metadata: a pandas dataframe containing detailed metadata for all samples

    """    

    files_dic = list_nanosight_files_in_directory(directory_path)
    

    # iterate over each file found in the directory
    for i, filename in enumerate(sorted(list(files_dic.keys()))):
        
        # store file infos to create file_infos pandas dataframe
        file_infos = []      
        
        # try to find dilution factor in filename
        # if dilution factor is not found, it is set to 1
        dilution_factor, has_dilution_been_found = get_dilution_infos(filename, dilution_prefix)
        
        if has_dilution_been_found:
            file_infos += [dilution_factor]
        else:
            file_infos += ['Not found']

        # try to find replicate group in filename, if yes get corresponding sample name
        # if replicate group is not found, filename is kept as sample name
        sample_name, has_replicate_group_been_found = get_replicate_group(filename, replicate_prefix)
        file_infos += [sample_name]
        
        file_infos = pandas.DataFrame([file_infos])
        file_infos.columns = ['Dilution factor', 'Sample name']
        file_infos.index = [filename]

        """ 
        read the file 'experiment summary' which contains the data
        
        """
        experiment_summary_file = files_dic[filename]["experiment_summary_file"]    
        experiment_summary_data = read_experiment_summary_file(Path(directory_path, experiment_summary_file))
        size_distributions, size_concentration_attributes, metadata = experiment_summary_data

        """ 
        multiply concentrations by the dilution factor and add info in experiment infos
        
        """        

        for col in [column for column in size_distributions.columns if "Bin centre" not in column]:
            size_distributions[col] = size_distributions[col] * dilution_factor 

        """ 
        calculate total concentration for each video
        
        """
        total_concentrations = ['Total concentration']
        videos_cols = [col for col in size_distributions.columns if 'Concentration Video' in col]
        for col in videos_cols:
            area = np.trapz(size_distributions[col], size_distributions['Bin centre (nm)'])
            total_concentrations.append(area)
        total_concentrations = pandas.DataFrame(np.array(total_concentrations).reshape(1,-1), 
                                                columns=['key']+[col.replace('Concentration ','') for col in videos_cols])
        size_concentration_attributes = pandas.concat([total_concentrations, size_concentration_attributes])
        size_concentration_attributes.reset_index(inplace=True, drop=True)

        """ 
        add average and standard deviation over all videos
        """
        
        # in size_distributions dataframe (at each bin center)
        videos_cols = [col for col in size_distributions.columns if 'Concentration Video' in col]
        size_distributions['Average'] = np.mean(size_distributions[videos_cols], axis=1)
        size_distributions['Std'] = np.std(size_distributions[videos_cols], axis=1)
        
        # in size_concentration_attributes dataframe
        size_concentration_attributes = size_concentration_attributes.apply(pandas.to_numeric, errors='ignore')
        size_concentration_attributes['Average'] = np.mean(size_concentration_attributes.iloc[:,1:], axis=1)
        size_concentration_attributes['Std'] = np.std(size_concentration_attributes.iloc[:,1:], axis=1)

        """
        reorganize dataframes to easily concatenate the results for all samples
        """

        # reorganize size concentration attributes dataframe to have only one row 
        # and a column for each combination of key info and video (or average/std over videos)
        # also add sample name as index

        size_concentration_attributes = pandas.DataFrame({f"{key} {col}": size_concentration_attributes[size_concentration_attributes['key'] == key][col].values 
                                          for key in size_concentration_attributes['key'].unique() 
                                          for col in size_concentration_attributes.columns if col != 'key'})
        size_concentration_attributes.index = [filename]

        # do the same for experiment_infos    
        metadata = pandas.DataFrame({f"{key} {col}": metadata[metadata['key'] == key][col].values 
                                          for key in metadata['key'].unique() 
                                          for col in metadata.columns if col != 'key'})
        metadata.index = [filename]
        
        # add filename in the column names of each concentration column
        size_distributions.columns = [col for col in size_distributions.columns + ' ' + filename]

        """
        concatenate results of new sample with others
        
        """

        if i==0:
            all_size_distributions = size_distributions.copy()
            all_size_concentration_attributes = size_concentration_attributes.copy()
            all_metadata = metadata.copy()
            all_files_infos = file_infos.copy()
            
        else:
            # for size_distributions, concatenation is in columns
            all_size_distributions = pandas.concat([all_size_distributions, size_distributions], axis=1)
            # for size concentration attributes and experiment infos, concatenation is in rows
            all_size_concentration_attributes = pandas.concat([all_size_concentration_attributes, size_concentration_attributes], axis=0)
            all_metadata = pandas.concat([all_metadata, metadata], axis=0)
            all_files_infos = pandas.concat([all_files_infos, file_infos], axis=0)

    """
    add a column summarizing particles per frame infos and noise infos over all videos
    
    """

    particles_per_frame = [all_metadata.loc[filename][[col for col in all_metadata.columns if 'Particles per frame Video' in col]].tolist()
                           for filename in all_metadata.index]

    noise_infos = [all_metadata.loc[filename][[col for col in all_metadata.columns if 'Noise level Video' in col]].tolist()
                           for filename in all_metadata.index]

    # defined the sample measure as reliable if 'OK' was written for all videos in nanosight exports
    no_noise_detected = [np.all(np.array(noise_infos[i]) == 'No') for i in range(len(noise_infos))]
    noise_infos = ['' if no_noise_detected[i] else 'Noise detected' for i in range(len(no_noise_detected))]

    all_metadata['Particles per frame'] = np.array(particles_per_frame).tolist()
    all_metadata['Noise detected'] = np.array(noise_infos)


    """
    verify the equality of bin center columns. Raise error if not. Then keep only one bin centre column
    
    """

    bin_centre_columns = [col for col in all_size_distributions.columns if 'Bin centre' in col]

    for i, j in list(itertools.combinations(np.arange(len(bin_centre_columns)), 2)):
        
        i_values = all_size_distributions[bin_centre_columns[i]].values
        j_values = all_size_distributions[bin_centre_columns[j]].values

        if (i_values==j_values).sum() != len(i_values):
            raise ValueError("Error: different bin sizes", directory_path)

    for col in bin_centre_columns[1:]:
        all_size_distributions.drop(col, axis=1, inplace=True)
        
    # rename remaining column to remove filename
    all_size_distributions.rename(columns={bin_centre_columns[0]: 'Bin centers'}, inplace=True)


    """
    create all_samples_filenames dict
    """
    
    # group all files infos by sample name
    all_samples_filenames = all_files_infos.groupby('Sample name').apply(lambda x: sorted(list(x.index))).to_dict()
    # transform dict in OrderedDict to have sorted samples
    all_samples_filenames = dict(sorted(all_samples_filenames.items()))
    # if there are no replicates for a sample, make the value an empty list
    # keep only replicate suffix in replicate filename
    all_samples_filenames = {sample_name: [filename.replace(sample_name,'') for filename in replicate_filenames] 
                             for sample_name, replicate_filenames in all_samples_filenames.items()}

    """
    add average size distributions and size concentration attributes over replicates if any
    """

    # iterate over samples and average data of all replicates
    for sample_name, replicates_filenames in all_samples_filenames.items():
        
        # if no replicate was detected for this sample, continue
        if replicates_filenames==[sample_name]:
            continue
        
        all_replicates_size_distributions = all_size_distributions[['Average '+sample_name+filename 
                                                                    for filename in replicates_filenames]]
        all_size_distributions['Average '+sample_name] = np.mean(all_replicates_size_distributions, axis=1)
        all_size_distributions['Std '+sample_name] = np.mean(all_replicates_size_distributions, axis=1)
        
        all_replicates_values = all_size_concentration_attributes.loc[[sample_name+filename 
                                                                       for filename in replicates_filenames]]
        
        
        all_size_concentration_attributes.loc[sample_name] = all_replicates_values.mean()
        
        # the average of videos values and the average of std over videos values makes no sense for samples, fill with NaN
        cols_to_write_nan = [col for col in all_size_concentration_attributes.columns if ('Video' in col or 'Std' in col)]
        all_size_concentration_attributes.loc[sample_name][cols_to_write_nan] = np.nan
        

    return {'files_infos': all_files_infos, 
            'samples_filenames': all_samples_filenames,
            'size_distributions': all_size_distributions, 
            'size_concentration_attributes': all_size_concentration_attributes, 
            'metadata': all_metadata
            }


def get_dilution_infos(filename, dilution_prefix):

          
    """
    try to find the dilution factor from sample name
        
        parameters
        ----------
        filename
        dilution_prefix: the dilution prefix to consider when reading filename
    
        returns
        ----------
        dilution_factor: dilution factor is found, else 1
        has_dilution_been_found: True if found, else False

    """  
    
    try:
        # get character string after dilution prefix
        dilution_chain = filename.split(dilution_prefix)[-1]
        # read number in dilution chain
        dilution_factor = ''
        for s in range(len(dilution_chain)):
            if dilution_chain[s].isdigit():
                dilution_factor += dilution_chain[s]  
            else:
                break
        dilution_factor = int(dilution_factor)            
        has_dilution_been_found = True

    except:
        has_dilution_been_found = False
        dilution_factor = 1 # default dilution factor
        
    return dilution_factor, has_dilution_been_found



def get_replicate_group(filename, replicate_prefix):


    """
    try to find if the export is a replicate measure of a sample
        
        parameters
        ----------
        filename
        replicate_prefix: the replicate prefix to consider when reading filename
        
    
        returns
        ----------
        sample name: sample name whose file is a measurement replicate
        has_replicate_group_been_found: True if replicate group has been found else False

    """     

    if replicate_prefix is None:
        
        has_replicate_group_been_found = False
        sample_name = filename

    else:
        has_replicate_group_been_found = replicate_prefix in filename
        
        if has_replicate_group_been_found:

            str_num = filename.split(replicate_prefix)[-1]

            sample_name = filename.replace(replicate_prefix+str_num,'')
            
        else:
            sample_name = filename
            
    return sample_name, has_replicate_group_been_found


        





    