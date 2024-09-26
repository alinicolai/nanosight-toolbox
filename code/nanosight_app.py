
import os
import tkinter
import tkinter.font as TkFont
from pathlib import Path
import numpy as np
import pandas
from scipy.integrate import simpson

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from paths import datapath, resultspath

from gui_params import bg_color, ratio_padx, ratio_pady

from app_tools.gui_tools import initialize_tkinter_graphical_interface, ask_data_directory, ask_and_store
from app_tools.other_tools import create_directory

from data_extraction_module.nanosight_data_extraction import extract_nanosight_data_from_directory

from data_analysis_module.plot_tools import plot_size_distributions, barplot
from data_analysis_module.clustering import run_wasserstein_clustering
from data_analysis_module.two_samples_tests import run_two_samples_tests


class NanosightApp():
        
    def __init__(self, 
                 mode='gui', 
                 chosen_directory='', 
                 dilution_prefix=None,
                 replicate_prefix=None):
                
        self.mode=mode
   
        # data directory (mandatory, default = current directory)
        self.chosen_directory = chosen_directory
        
        # prefix preceding the sample dilution (optional)
        self.dilution_prefix = dilution_prefix
        
        # prefix preceding the replicate number if any (optional)
        self.replicate_prefix = replicate_prefix

        # will store data exports
        self.data = None
        
        # indicates if there exist replicates of the sample sample, will be defined after loading data
        self.any_replicates = None
        # will be defined after loading data
        self.filenames = None 
        # samples names can be different of filenames if there exist replicates of the same sample
        self.samples_names = None 
        
        # can be defined by the user via the gui, optional (allows to run two-samples tests)
        self.sample_class_labels = None
        self.files_class_labels = None


    def run(self):
        
        """
        run the application
        
        """        
        
        if self.mode=='manual':
            # manual mode is usefull for testing and debugging
            self.extract_data()
            self.export_data()
            self.run_data_analysis()
            
        elif self.mode=='gui':
            # creates the graphical user interface
            self.run_gui()


    def run_gui(self):

        """
        create the graphical user interface
        
        """ 
        
        # initialize the gui using the function in app_tools.py
        self.gui_root = initialize_tkinter_graphical_interface(bg_color=bg_color,
                                                            title='Nanosight data analysis') 

        # create a frame dedicated to data loading, attached to self.gui_root created at the previous step
        self.load_data_frame = tkinter.LabelFrame(self.gui_root, 
                                                  text="Load data", 
                                                  bg=bg_color,
                                                  font = TkFont.Font(weight="bold"))
        
        # place this frame at row 0 and column 0 in self.gui_root
        # padx: if a single value is provided, the same amount of horizontal space is added on both the left and right sides of the frame
        # pady: if a single value is provided, the same amount of horizontal space is added on both the top and bottom of the frame
        # sticky = 'nsw': extends the widget to occupy all the available space in the cell (top, bottom, left, and right).
        self.load_data_frame.grid(row = 0, column = 0, pady=20*ratio_pady, padx=20*ratio_padx, sticky="nswe")
        

        """
        ask the user for the data directory path and save it
        
        """
        
        # create tkinter var of type String to store the data directory path
        self.chosen_directory_tkinter_var = tkinter.StringVar()

        # create a button 'choose directory' attached to self.load_data_frame
        # when the button is clicked, it runs the function ask_data_directory (in app_tools.gui_tools)
        button_chose_path = tkinter.Button(self.load_data_frame, 
                                           text = "Choose directory", 
                                           command = lambda: ask_data_directory(self.chosen_directory_tkinter_var, 
                                                                                initial_directory=datapath), 
                                           bg=bg_color, 
                                           fg="black")
        
        # place this button at row 1, column 1 in self.load_data_frame
        button_chose_path.grid(column=1, row=1, pady=40*ratio_pady, padx=10*ratio_padx)
        
        # display the name of the chosen directory, at row 1, column 2
        self.chosen_directory_display = tkinter.Label(self.load_data_frame, 
                                                      textvariable=self.chosen_directory_tkinter_var, 
                                                      background=bg_color, fg="orangered")
        self.chosen_directory_display.grid(column=2, row=1, pady=40*ratio_pady, padx=50*ratio_padx)

        # store user entry at each modification in the class attribute self.chosen_directory
        self.chosen_directory_tkinter_var.trace_add(mode='write', 
                                                   callback=self.on_chosen_directory_change)

        """
        ask the user for the dilution prefix and store it
        
        """ 

        # create tkinter var of type String to store the tnered dilution prefix
        self.dilution_prefix_tkinter_var = tkinter.StringVar(self.load_data_frame)

        # place user entry at row 2, column 2 and its title at row 2, column 1
        ask_and_store(frame=self.load_data_frame,
                      tkinter_var=self.dilution_prefix_tkinter_var,
                      title='Dilution prefix', 
                      label_position=[2,1], 
                      entry_position=[2,2], 
                      default_value='dilution',
                      bg_color=bg_color,
                      ratio_pady=ratio_pady)
        
        # store user entry at each modification in the class attribute self.dilution_prefix
        self.dilution_prefix_tkinter_var.trace_add(mode='write', 
                                                   callback=self.on_dilution_entry_change)

        """
        ask the user for the replicate prefix and store it
        
        """ 
        # create tkinter var of type String to store the tnered dilution prefix
        self.replicate_prefix_tkinter_var = tkinter.StringVar(self.load_data_frame)

        # place user entry at row 3, column 2 and its title at row 3, column 1 and u
        ask_and_store(frame=self.load_data_frame,
                      tkinter_var=self.replicate_prefix_tkinter_var,
                      title='Replicate prefix',
                      label_position=[3,1], 
                      entry_position=[3,2], 
                      default_value='',
                      bg_color=bg_color,
                      ratio_pady=ratio_pady)

        # store user entry at each modification in the class attribute self.replicate_prefix
        self.replicate_prefix_tkinter_var.trace_add(mode='write', 
                                                   callback=self.on_replicate_entry_change)

        """
        add a button for data loading; when clicked this button runs the function self.load_data
        
        """ 
        
        # create a button 'Load'; when clicked this runs the function self.execute_workflow that will process and display data
        button_export_nanosight = tkinter.Button(self.load_data_frame, text = 'Load', command = self.execute_workflow, bg="white", fg="black")
        button_export_nanosight.grid(row=5, columnspan=3, column=0, pady=40*ratio_pady)

        tkinter.mainloop()


    def execute_workflow(self):
        
        """
        extract data and run all consecutive actions
        
        """ 

        # extract_data
        self.extract_data()
        
        # display export_infos (samples list and ather relevant information)
        self.display_export_infos()
        
        # propose to the user to add labels 
        self.propose_to_add_labels()
        
        # propose data analysis options
        self.propose_analysis_options()

        # adjust canvas size to limit height to 0.8 x screen height and manage other settings
        self.adjust_canvas_frame()
        
    
            

    def on_chosen_directory_change(self, *args):
        
        """
        at each modification of the data directory by the user, store it in self.chosen_directory
        
        """
        self.chosen_directory = self.chosen_directory_tkinter_var.get()

        # remove any previously loaded objects, as an export parameter has changed.
        # rhe user will need to click again on 'Load' to reload the data
        self.reset_data()


    def on_dilution_entry_change(self, *args):
        
        """
        at each modification of dilution prefix by the user, store it in self.dilution_prefix

        """

        self.dilution_prefix = self.dilution_prefix_tkinter_var.get()
        
        # make sure an empty entry is not considered as a true String prefix
        if self.dilution_prefix =='':
            self.dilution_prefix = None

        # remove any previously loaded objects, as an export parameter has changed.
        # rhe user will need to click again on 'Load' to reload the data
        self.reset_data()
    
    
    def on_replicate_entry_change(self, *args):
        
        """
        at each modification of replicate prefix by the user, store it in self.replicate_prefix
        
        """
        
        self.replicate_prefix = self.replicate_prefix_tkinter_var.get()

        # make sure an empty entry is not considered as a true String prefix        
        if self.replicate_prefix == '':
            self.replicate_prefix = None

        # remove any previously loaded objects, as an export parameter has changed.
        # rhe user will need to click again on 'Load' to reload the data        
        self.reset_data()


    def reset_data(self):
        
        # reset data attribute
        self.data = None

        # clear old data display on the gui if any

        if hasattr(self, 'list_samples_frame'):
            self.list_samples_frame.destroy()

        if hasattr(self, 'analysis_frame'):
            self.analysis_frame.destroy()
            
        if hasattr(self, "data_correctly_loaded"):
            self.data_correctly_loaded.destroy()
            

    def extract_data(self):
        
        """
        extract all data from self.chosen_directory
        it uses the export settings stored in the class attributes (dilution_prefix, replicate_prefix)
        in manual mode, these settings are provided during class instantiation
        in gui mode, the user is asked to provide these settings

        store in self.data
        ----------
        a dictionary containing: 
        files_infos: a pandas dataframe containing files information (replicates if any, samples, dilutions)
        samples_filenames: a dictionary where the keys are sample names and the values are the corresponding list of replicates filenames
        size_distributions: a pandas dataframe containing size_distributions for all samples
        size_concentration_attributes: a pandas dataframe containing all size and concentration attributes for all samples
        metadata: a pandas dataframe containing detailed metadata for all samples
        
        """

        self.data = extract_nanosight_data_from_directory(directory_path=Path(datapath, self.chosen_directory),
                                                             dilution_prefix=self.dilution_prefix,
                                                             replicate_prefix=self.replicate_prefix)

        self.filenames = self.data['files_infos'].index
        
        # check if replicates exist for at least one sample and store the information
        # the list of replicates must be of length > 1 for at least one sample
        self.any_replicates = sum([True if len(v)>1 else False for k,v in self.data['samples_filenames'].items()])>0
        
        self.samples_names = list(self.data['samples_filenames'].keys())


    def display_export_infos(self):
        
        """
        display samples list and other relevant information (dilution, replicates, particules per frame...)
        
        """ 

        # indicate that data have been correctly loaded
        self.data_correctly_loaded = tkinter.Label(self.load_data_frame, text = "Data correctly loaded", bg=bg_color, fg="orangered")
        self.data_correctly_loaded.grid(row=7, columnspan=3, column=0, pady=10*ratio_pady)

        # create a frame at the right of the load data frame, to display sample list
        self.list_samples_frame = tkinter.LabelFrame(self.gui_root, text="List of samples", font = TkFont.Font(weight="bold"), bg=bg_color)
        self.list_samples_frame.grid(row=0, column=1, padx=40*ratio_padx, pady=20*ratio_pady, sticky='nswe')

        # add a canvas in that frame, (mandatory to add a scrollbar later)
        # the scrollbar will help to see all samples if the list is too long to fully appear on the screen
        self.canvas = tkinter.Canvas(self.list_samples_frame, bg=bg_color)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        
        # create a window inside canvas to add content (mandatory)
        self.canvas_window = tkinter.Frame(self.canvas, bg=bg_color)
        self.canvas.create_window((0, 0), window=self.canvas_window, anchor='nw')

        # link a scrollbar to the canvas frame
        vsb = tkinter.Scrollbar(self.list_samples_frame, orient="vertical", command=self.canvas.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        #synchronizes the canvas with the vertical scrollbar so they work together
        self.canvas.config(yscrollcommand=vsb.set, bg=bg_color)

        # add column titles for display: 'Name', ('Replicate name'), 'Dilution', 'Particles per frame', 'Noise detected'
        names_title = tkinter.Label(self.canvas_window, text = "Name", bg=bg_color, fg="black")
        names_title.grid(row=0, column=0, pady=40*ratio_pady)
        index_col = 0

        # if replicates exist, display a column title for replicates names
        if self.any_replicates:
            replicate_names_title = tkinter.Label(self.canvas_window, text = 'Replicate name', bg=bg_color, fg="black")
            replicate_names_title.grid(row=0, column=1, pady=40*ratio_pady, padx=2*ratio_padx, sticky='nswe')
            index_col += 1
                         
        dilutions_title = tkinter.Label(self.canvas_window, text = 'Dilution', bg=bg_color, fg="black")
        dilutions_title.grid(row=0, column=index_col+1, pady=40*ratio_pady, padx=2*ratio_padx, sticky='nswe')

        particles_per_frame_title = tkinter.Label(self.canvas_window, text = 'Particles per frame', bg=bg_color, fg="black")
        particles_per_frame_title.grid(row=0, column=index_col+2, pady=40*ratio_pady, padx=2*ratio_padx, sticky='nswe')

        validity_title = tkinter.Label(self.canvas_window, text = 'Noise detected', bg=bg_color, fg="black")
        validity_title.grid(row=0, column=index_col+3, pady=40*ratio_pady, padx=2*ratio_padx, sticky='nswe')
      
        # iterate over samples and display all information
        
        sample_row_index = 1

        for i, (sample_name, replicate_filenames) in enumerate(self.data['samples_filenames'].items()):

            sample_name_label = tkinter.Label(self.canvas_window, text = sample_name, bg=bg_color, fg="black")
            sample_name_label.grid(row=sample_row_index, column=0, pady=2*ratio_pady)

            if self.any_replicates:
                           
                for r, replicate_filename in enumerate(replicate_filenames):

                    files_infos = self.data['files_infos'].loc[sample_name+replicate_filename]
                    metadata = self.data['metadata'].loc[sample_name+replicate_filename]

                    replicate_name_label = tkinter.Label(self.canvas_window, text = replicate_filename, bg=bg_color, fg="black")
                    replicate_name_label.grid(row=sample_row_index+r, column=1, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
      
                    dilution_label = tkinter.Label(self.canvas_window, text = str(files_infos['Dilution factor']), bg=bg_color, fg="black")
                    dilution_label.grid(row=sample_row_index+r, column=2, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
            
                    particle_per_frame_label = tkinter.Label(self.canvas_window, text = ', '.join(metadata['Particles per frame']), bg=bg_color, fg="black")
                    particle_per_frame_label.grid(row=sample_row_index+r, column=3, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
            
                    valid_label = tkinter.Label(self.canvas_window, text = metadata['Noise detected'], bg=bg_color, fg="black")
                    valid_label.grid(row=sample_row_index+r, column=4, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
            
                sample_row_index += r+1
    
            else:
                files_infos = self.data['files_infos'].loc[sample_name]

                metadata = self.data['metadata'].loc[sample_name]

                dilution_label = tkinter.Label(self.canvas_window, text = str(files_infos['Dilution factor']), bg=bg_color, fg="black")
                dilution_label.grid(row=sample_row_index, column=1, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
        
                particle_per_frame_label = tkinter.Label(self.canvas_window, text = ', '.join(metadata['Particles per frame']), bg=bg_color, fg="black")
                particle_per_frame_label.grid(row=sample_row_index, column=2, pady=10*ratio_pady, padx=2*ratio_padx, sticky='nswe')
        
                valid_label = tkinter.Label(self.canvas_window, text = metadata['Noise detected'], bg=bg_color, fg="black")
                valid_label.grid(row=sample_row_index, column=3, pady=2*ratio_pady, padx=10*ratio_padx, sticky='nswe')
        
                sample_row_index +=1

        self.adjust_canvas_frame()


    def propose_to_add_labels(self):

        self.add_labels_button = tkinter.Button(self.canvas_window, text = 'Add class label', command=self.ask_labels, bg='white')
        self.add_labels_button.grid(row=0, column=5, pady=10*ratio_pady, padx=30*ratio_padx)
        
        
    def ask_labels(self):
        
        self.add_labels_button.destroy()

        self.label_title = tkinter.Label(self.canvas_window, text = 'Enter class label (0 or 1)', bg='white')
        self.label_title.grid(row=0, column=5, pady=10*ratio_pady, padx=30*ratio_padx)
        
        # all are set to 0 by default
        self.samples_class_labels_tkinter_vars = [tkinter.StringVar(value='0') for i in range(len(self.samples_names))]
        
        self.samples_class_labels_entries = []
        
        
        row_index = 1
        
        for i, sample_name in enumerate(self.samples_names):
            
            nb_replicates = len(self.data['samples_filenames'][sample_name])
            
            label_entry = tkinter.Entry(self.canvas_window, textvariable=self.samples_class_labels_tkinter_vars[i], width=3)
            
            label_entry.grid(row=row_index, column=5, pady=2*ratio_pady, padx=10*ratio_padx)

            self.samples_class_labels_entries.append(label_entry)
            
            row_index += nb_replicates

        self.ok_labels_button = tkinter.Button(self.canvas_window, text = 'Ok', command=self.store_labels, bg='white')
        self.ok_labels_button.grid(row=len(self.filenames)+1, column=5, pady=2*ratio_pady, padx=10*ratio_padx)

        self.adjust_canvas_frame()
       
        
    def store_labels(self):
        
        # class_labels will be filled with the values entered by the user
        self.samples_class_labels = []

        # fill with the values entered by the user
        for i in range(len(self.samples_names)):

            # only if 0 or 1
            if (self.samples_class_labels_tkinter_vars[i].get()!='0' and self.samples_class_labels_tkinter_vars[i].get()!='1'):
            
                return
            
            self.samples_class_labels.append(int(self.samples_class_labels_tkinter_vars[i].get()))

            # class_label_entries[i] is not necessary anymore and will be created again if needed
            self.samples_class_labels_entries[i].destroy()
        
        # change column title to indicate that class labels have been entered
        self.label_title.destroy()
        self.label_title = tkinter.Label(self.canvas_window, text = 'Class labels', bg=bg_color)
        self.label_title.grid(row=0, column=5, pady=2*ratio_pady, padx=10*ratio_padx)

        # show the user the class labels that have been stored
        self.samples_class_labels_displays = []        
        
        row_index = 1
        for i, sample_name in enumerate(self.samples_names):
            
            nb_replicates = len(self.data['samples_filenames'][sample_name])

            label_display = tkinter.Label(self.canvas_window, text=str(self.samples_class_labels[i]), bg=bg_color)
            label_display.grid(row=row_index, column=5, pady=2*ratio_pady, padx=10*ratio_padx)
            
            row_index += nb_replicates
                    
            self.samples_class_labels_displays.append(label_display)

        # destroy the button to validate labels and replace it by a remove button to erase entered class labels
        self.ok_labels_button.destroy()
        self.remove_labels_button = tkinter.Button(self.canvas_window, text = 'Remove', command=self.remove_labels, bg='white')
        self.remove_labels_button.grid(row=len(self.filenames)+1, column=5, pady=2*ratio_pady, padx=10*ratio_padx)

        # now that labels have been stored, propose to run two-samples tests
        self.button_tests = tkinter.Button(self.analysis_frame, text = 'Two-sample tests', command = self.run_two_samples_tests, bg='white', fg='black')
        self.button_tests.grid(row=3, column=0, pady=40*ratio_pady, padx=20*ratio_padx)    
        
        # labels have been defined for each sample, we need to expand the information to all replicates
        # it will be usefull if we dont want to group replicates in two-samples tests
        self.files_class_labels = []
        for i, filename in enumerate(self.filenames):
            corresponding_sample_name = self.data['files_infos'].loc[filename]['Sample name']
            index_sample_name = self.samples_names.index(corresponding_sample_name)
            class_label = self.samples_class_labels[index_sample_name]
            self.files_class_labels.append(class_label)
            
        self.adjust_canvas_frame()
                
        
    def remove_labels(self):
        
        self.label_title.destroy()
    
        if hasattr(self, 'ok_tests'):
            self.ok_tests.destroy()

        for i in range(len(self.samples_names)):
            self.samples_class_labels_displays[i].destroy()
            
        self.remove_labels_button.destroy()
        
        self.propose_to_add_labels()
        
        self.button_tests.destroy()

        self.adjust_canvas_frame()
        
        self.samples_class_labels = None
              

    def propose_analysis_options(self):

        # create frame for data analysis
        self.analysis_frame = tkinter.LabelFrame(self.gui_root, text="Analysis", font = TkFont.Font(weight="bold"),
                                                 bg = bg_color)
        self.analysis_frame.grid(row=0, column=2, sticky='nw', padx=20*ratio_padx, pady=20*ratio_pady)

        # export button, when clicked the extracted data are exported in csv files in the results directory
        button_export = tkinter.Button(self.analysis_frame, text = 'Data csv export' , command = self.export_data, bg='white', fg='black')
        button_export.grid(row=0, column=0, pady=40*ratio_pady, padx=20*ratio_padx)
        
        button_plot = tkinter.Button(self.analysis_frame, text = 'Data illustrations' , command = self.plot, bg='white', fg='black')
        button_plot.grid(row=1, column=0, pady=40*ratio_pady, padx=20*ratio_padx)

        button_clustering = tkinter.Button(self.analysis_frame, text = 'Clustering' , command = self.run_clustering, bg='white', fg='black')
        button_clustering.grid(row=2, column=0, pady=40*ratio_pady, padx=20*ratio_padx)        


    def run_data_analysis(self):
        
        # only used in manual mode
        self.export_data()
        self.plot()
        self.run_clustering()



    def export_data(self):
        
        """
        export all self.data in elements in csv format in the results directory
        
        """ 
        
        if self.mode=='gui':
            # remove the confirmation of previous export if any
            if hasattr(self, 'ok_export'):
                self.ok_export.destroy()

        # create a directory for csv exports
        create_directory([resultspath, self.chosen_directory, 'data_csv_export'])
        csv_savepath = os.path.join(resultspath, self.chosen_directory, 'data_csv_export')
        
        # export all self.data elements in csv format
        for key in self.data:
            if isinstance(self.data[key], pandas.DataFrame):
                # save index in the csv only if it contains information
                if type(self.data[key].index[0])==str:
                    index = True
                else:
                    index = False
                self.data[key].to_csv(os.path.join(csv_savepath, key+'.csv'), index=index)

        if self.mode=='gui': 
            # display 'Ok' when export is successfull
            self.ok_export = tkinter.Label(self.analysis_frame, text = "Ok", bg=bg_color, fg="orangered")
            self.ok_export.grid(row=0, column=1, pady=40*ratio_pady, padx=20*ratio_padx)
            


    def plot(self):
        
        """
        generate all plots
        
        """ 

        # remove the confirmation of previous plot if any
        if self.mode == 'gui':
            if hasattr(self, 'ok_plot'):
                self.ok_plot.destroy()

        # create a directory for csv exports
        create_directory([resultspath, self.chosen_directory, 'data_illustrations'])
        
        # generate plots
        self.plot_size_distributions()
        self.plot_size_concentration_attributes()

        # display 'Ok' when export is successfull
        if self.mode=='gui': 
            self.ok_plots = tkinter.Label(self.analysis_frame, text = "Ok", bg=bg_color, fg="orangered")
            self.ok_plots.grid(row=1, column=1, pady=40*ratio_pady, padx=20*ratio_padx)
                    
        
    def plot_size_distributions(self):

        plots_savepath = os.path.join(resultspath, self.chosen_directory, 'data_illustrations')

        bin_centers = self.data['size_distributions']['Bin centers'].values
        
        """
        generate plots for all files (plot all video distribs and average/std)
        
        """ 
        for filename in self.filenames:
            videos_cols = [col for col in self.data['size_distributions'].columns if filename in col and ' Video' in col]
            list_concentrations = [self.data['size_distributions'][col] for col in videos_cols]
            list_legend_labels = [col.replace(filename,'').replace('Concentration ','') for col in videos_cols]
            
            plot_size_distributions(bin_centers, list_concentrations, savepath=plots_savepath, 
                                       list_legend_labels=list_legend_labels, name=filename, title=filename)

        # all file average distributions in one plot
        list_concentrations = [self.data['size_distributions']['Average '+filename] for filename in self.filenames]
        list_legend_labels = self.filenames
        plot_size_distributions(bin_centers, list_concentrations, savepath=plots_savepath, 
                                    list_legend_labels=list_legend_labels, name='all_files')

        # all file average distributions in one plot + normalize distributions (divide by area to have densities)
        list_concentrations = [self.data['size_distributions']['Average '+filename] for filename in self.filenames]
        list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
        list_legend_labels = [filename+', normalized' for filename in self.filenames]
        plot_size_distributions(bin_centers, list_normalized_concentrations, savepath=plots_savepath, 
                                    list_legend_labels=list_legend_labels, name='all_files_normalized')

        """
        if replicates exist, plot replicates distributions and average/std for all samples
        
        """
        # 
        if self.any_replicates:
                        
            # plot individual distributions
            for sample_name in self.samples_names:
                
                list_replicates = self.data['samples_filenames'][sample_name]

                # check if there are at least two replicates, otherwise continue
                if len(list_replicates) < 2 :
                    continue

                replicate_average_concentrations = [self.data['size_distributions']['Average '+sample_name+filename] 
                                                                                   for filename in list_replicates]
                
                plot_size_distributions(bin_centers, replicate_average_concentrations, savepath=plots_savepath, 
                                            list_legend_labels=list_replicates, name=sample_name, title=sample_name)

            # plot all samples average distributions
            list_concentrations = [self.data['size_distributions']['Average '+sample_name] 
                                           for sample_name in self.samples_names if len(self.data['samples_filenames'][sample_name])>=2]
            list_legend_labels = [sample_name for sample_name in self.samples_names if len(self.data['samples_filenames'][sample_name])>=2]
            plot_size_distributions(bin_centers, list_concentrations, list_legend_labels=list_legend_labels, name='all_samples', 
                                        savepath=plots_savepath)

            # plot all samples average distributions + normalize distributions (divide by area to have densities)
            list_concentrations = [self.data['size_distributions']['Average '+sample_name] 
                                           for sample_name in self.samples_names if len(self.data['samples_filenames'][sample_name])>=2]
            list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
            list_legend_labels = [sample_name+', normalized' for sample_name in self.samples_names if len(self.data['samples_filenames'][sample_name])>=2]
            plot_size_distributions(bin_centers, list_normalized_concentrations, list_legend_labels=list_legend_labels, name='all_samples_normalized', 
                                        savepath=plots_savepath)



    def plot_size_concentration_attributes(self):

        plots_savepath = os.path.join(resultspath, self.chosen_directory, 'data_illustrations')
        
        """
        bar plots for each attribute across all files
        
        """ 
                
        list_attributes = [col.replace(' Average','') for col in self.data['size_concentration_attributes'].columns if 'Average' in col]
        
        for attribute in list_attributes:

            attribute_data = self.data['size_concentration_attributes'].loc[self.filenames][attribute+' Average'].values
            
            barplot(attribute_data, name='all_files_'+attribute, list_legend_labels = self.filenames, 
                    savepath=plots_savepath, title=attribute)


        """
        bar plots for each attribute across all samples (if replicates exist)
        
        """ 
        
        if self.any_replicates:
                
            for attribute in list_attributes:
    
                attribute_data = self.data['size_concentration_attributes'].loc[self.samples_names][attribute+' Average'].values

                barplot(attribute_data, name='all_samples_'+attribute, list_legend_labels = list(self.samples_names), 
                        savepath=plots_savepath, title=attribute)




    def run_clustering(self):
        
        """
        run clustering of size distributions and size concentration attributes
        
        """ 
        
        if self.mode=='gui':
            # remove the confirmation of previous export if any
            if hasattr(self, 'ok_clustering'):
                self.ok_clustering.destroy()

        # create a directory for clustering exports
        create_directory([resultspath, self.chosen_directory, 'clustering'])
        clustering_savepath = os.path.join(resultspath, self.chosen_directory, 'clustering')

        """
        run clustering of all files size distributions
        
        """ 
        
        bin_centers = self.data['size_distributions']['Bin centers'].values

        list_concentrations = [self.data['size_distributions']['Average '+filename] for filename in self.filenames]
        list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
        list_legend_labels = self.filenames        

        run_wasserstein_clustering(bin_centers, list_normalized_concentrations, list_legend_labels, name='all_files', 
                                   savepath=clustering_savepath)
        
        
        
        """
        run clustering of all samples size distributions (if replicates exist)
        
        """ 

        if self.any_replicates:

            list_concentrations = [self.data['size_distributions']['Average '+sample_name] for sample_name in self.samples_names]
            list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
            list_legend_labels = self.samples_names
    
            run_wasserstein_clustering(bin_centers, list_normalized_concentrations, list_legend_labels, name='all_samples', 
                                       savepath=clustering_savepath)
            

        if self.mode=='gui': 
            # display 'Ok' when export is successfull
            self.ok_clustering = tkinter.Label(self.analysis_frame, text = "Ok", bg=bg_color, fg="orangered")
            self.ok_clustering.grid(row=2, column=1, pady=40*ratio_pady, padx=20*ratio_padx)
            
            
    def run_two_samples_tests(self):
        
        """
        run two_samples_tests of size distributions and size concentration attributes
        
        """ 

        if self.mode=='gui':
            # remove the confirmation of previous export if any
            if hasattr(self, 'ok_tests'):
                self.ok_tests.destroy()

        # create a directory for clustering exports
        create_directory([resultspath, self.chosen_directory, 'two-samples test'])
        tests_savepath = os.path.join(resultspath, self.chosen_directory, 'two-samples tests')

        """
        run two-samples tests between classes, considering all files size distributions
        
        """ 
        
        bin_centers = self.data['size_distributions']['Bin centers'].values

        list_concentrations = [self.data['size_distributions']['Average '+filename] for filename in self.filenames]
        list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
        list_legend_labels = self.filenames        

        run_two_samples_tests(bin_centers, list_normalized_concentrations, list_legend_labels, 
                                   list_class_labels=self.files_class_labels, name='all_files', 
                                   savepath=tests_savepath)

        """
        run two-samples tests between classes, considering all samples size distributions
        
        """ 
        
        bin_centers = self.data['size_distributions']['Bin centers'].values

        list_concentrations = [self.data['size_distributions']['Average '+filename] for filename in self.samples_names]
        list_normalized_concentrations = [distrib / simpson(x=bin_centers, y=distrib) for distrib in list_concentrations]
        list_legend_labels = self.samples_names        

        run_two_samples_tests(bin_centers, list_normalized_concentrations, list_legend_labels, 
                                   list_class_labels=self.samples_class_labels, name='all_files', 
                                   savepath=tests_savepath)

        if self.mode=='gui': 
            # display 'Ok' when export is successfull
            self.ok_tests = tkinter.Label(self.analysis_frame, text = "Ok", bg=bg_color, fg="orangered")
            self.ok_tests.grid(row=3, column=1, pady=40*ratio_pady, padx=20*ratio_padx)
            
            
    def adjust_canvas_frame(self):

        # # ensure all content is rendered before updating the scroll region
        self.canvas_window.update_idletasks()
        
        # ensure the canvas window columns can expand to prevent label content from being hidden
        max_column_index = max(widget.grid_info()['column'] for widget in self.canvas_window.winfo_children())

        for i in range(max_column_index+1):

            self.canvas_window.grid_columnconfigure(i, weight=1)  # First column
            
            
    
        # get the required width and height of the content inside the canvas
        content_width = self.canvas_window.winfo_reqwidth() 
        content_height = self.canvas_window.winfo_reqheight()
    
        # set a maximum height for the canvas (e.g., 50% of the screen height)
        max_height = 0.5 * self.gui_root.winfo_screenheight()
    
        # Update canvas size: use full width of the content but limit height
        self.canvas.config(width=content_width, height=min(content_height, max_height))
    
        # Configure the canvas scrolling region based on the content size
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
