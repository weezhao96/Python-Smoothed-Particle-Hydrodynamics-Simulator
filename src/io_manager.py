#%% IO Manager

#%% Main Class

class IO_Manager(object):
    
    # Type Annotation
    output_folder: str

    def __init__(self, output_folder: str):
    
        # Output Folder
        self.output_folder = output_folder