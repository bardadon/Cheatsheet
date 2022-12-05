def print_class_methods(b1):

    method_list = []
    for method in dir(b1):
        if '__' not in method:
            method_list.append(method)

    return method_list

def pipFreeze_RemoveVersion(temp_file, requirements_file = 'requirements.txt'):
    
    '''
    Remove version requirements from pip freeze.
    Create a file using pip freeze before running this function.
    ------------------Pip freeze >> temp.txt------------------

    Args:
        - temp_file = text file
    Returns:
        - None
    '''

    with open(temp_file, 'r') as read_file:

        with open(requirements_file, 'w') as write_file:

            for line in read_file:

                start_location = line.find('=')
                line = line[0:start_location] + '\n'
                
                write_file.writelines(line)

