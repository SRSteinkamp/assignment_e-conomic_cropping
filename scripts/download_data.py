import os
from cropping_lib.utils import get_data, check_working_dir

# Check if script is executed in correct location
BASEPATH = check_working_dir(os.path.realpath(__file__))

URL = 'https://github.com/e-conomic/hiring-assignments/raw/master/machinelearningteam/receipt-cropping/cropping_data.zip'
# Download data
get_data(URL, BASEPATH + '/cropping_data')
