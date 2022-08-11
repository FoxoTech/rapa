import pytest
import rapa

# test rapa.base.RAPABase initialization
def test_RAPABase_initialization():
    '''Checks that the rapa.base.RAPABase objects can't be initialized directly, and 
    should be initialized with a child class.

    1. Tests that the rapa.base.RAPABase class cannot be initialized directly
    2. Tests that the rapa.base.RAPABase can be initialized through a child class
    '''

    # 1. Tests that the rapa.base.RAPABase class cannot be initialized directly
    try:
        test_object = rapa.base.RAPABase()
    except RuntimeError:
        # this is expected
        pass
    else:
        raise Exception("`rapa` base class was initialied directly.")
        

        
