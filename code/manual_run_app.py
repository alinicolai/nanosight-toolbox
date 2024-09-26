


from paths import datapath
from nanosight_app import NanosightApp

    

directory = 'data directory'


app = NanosightApp(mode='manual', chosen_directory=directory, dilution_prefix='D', replicate_prefix='rep')

app.run()



