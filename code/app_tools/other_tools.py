import os
from pathlib import Path





def create_directory(list_sequential_dirs):
    
    for i in range(len(list_sequential_dirs)):
        
        path = Path(*list_sequential_dirs[:i+1])

        if not os.path.exists(path):
            os.mkdir(Path(path))
