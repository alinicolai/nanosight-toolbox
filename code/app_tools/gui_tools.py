
import os
import tkinter
from tkinter import filedialog



def initialize_tkinter_graphical_interface(bg_color, title):
  
    # create Tkinter interface
    gui = tkinter.Tk()
    
    # disables window resizing, preventing the user from changing the width or height of the main window.
    gui.resizable(width=False, height=False)        

    # display a title
    gui.title(title)
    
    # configure background color of the gui
    gui.configure(background=bg_color)            
        
    return gui




def ask_and_store(frame, tkinter_var, label_position, entry_position, title, bg_color, 
                  default_value, entry_size=10, ratio_pady=1):

    """
    Creates an input field for the user to enter an information
    Returns: the entered value
    """ 
    # indicate to the user that the information must be entered
    tkinter_label = tkinter.Label(frame, text=title, bg=bg_color, fg="black")
    # place it at the required position
    tkinter_label.grid(column=label_position[1], row=label_position[0], pady=40*ratio_pady)

    # create an input field for the user to enter the information
    dilution_entry = tkinter.Entry(frame, textvariable=tkinter_var, width=entry_size)
    # place it at row 2 column 2
    dilution_entry.grid(column=entry_position[1], row=entry_position[0])#, pady=40*ratio_pady)

        

             
   
def ask_data_directory(choosen_directory_tkinter_var, initial_directory):
    
    title = 'Choose data directory'
    entry = tkinter.filedialog.askdirectory(title=title, initialdir=initial_directory)

    choosen_directory_tkinter_var.set(os.path.relpath(entry, initial_directory))
    
    