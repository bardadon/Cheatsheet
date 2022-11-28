import pytest
from Helper import print_class_methods

def test_PrintClassMethods_CanReturnList():

    test_input = dict()
    test_list = print_class_methods(test_input)

    assert len(test_list) > 0



