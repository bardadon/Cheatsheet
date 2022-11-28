def print_class_methods(b1):

    method_list = [method for method in dir(b1) if '__' not in method]
    return method_list
