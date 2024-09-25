
from pathlib import Path
import os





source = Path().resolve().parent




datapath = Path(source, "data")
codepath = Path(source, "code")




for dir_ in ["nanosight_app_results"]:
    if not os.path.exists(Path(source, dir_)):
        os.mkdir(Path(source, dir_))

resultspath = Path(source, "nanosight_app_results")
