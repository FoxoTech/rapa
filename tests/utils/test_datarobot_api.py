import pytest

import rapa
import os
import pickle


# test api initialization
def test_api_initialization():
    '''Checks that the api can be connected with `utils.initialize_dr_api`.

    Currently only checks the default endpoint: https://app.datarobot.com/api/v2.
    '''
    print('test_api_inizialization test called')

    pkl_file_name = 'dr-tokens.pkl'
    dr_test_api_key = os.environ.get('DR_TEST_RAPA') # get the api key

    ## create the pickle file
    pickle.dump({'test':dr_test_api_key}, open(pkl_file_name, 'wb'))

    try:
        retval = rapa.utils.initialize_dr_api('test', pkl_file_name)
    except ValueError:
        ## delete the pickle file
        os.remove(pkl_file_name) 
        raise ValueError("API Key is Incorrect, check that the key is still valid in DataRobot.")

    ## delete the pickle file
    os.remove(pkl_file_name) 

    assert retval == None
