def print_class_methods(b1):

    method_list = []
    for method in dir(b1):
        if '__' not in method:
            method_list.append(method)

    return method_list

