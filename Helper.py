def print_class_methods(b1):

    for method in dir(b1):
        if '__' not in method:
            print(method)