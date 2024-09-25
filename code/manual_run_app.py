


from paths import datapath
from nanosight_app import NanosightApp

    

directory = 'Data Lea NTA Videodrop'


app = NanosightApp(mode='manual', chosen_directory=directory, dilution_prefix='D', replicate_prefix='rep')

app.run()



