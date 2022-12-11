###         ###
### General ###
###         ###

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





###         ### 
### Google  ###
###         ###
import os 
from google.cloud import bigquery
import google

# Creating an Environmental Variable for the service key configuration
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ServiceKey_GoogleCloud.json'

# Create a client
bigquery_client = bigquery.Client()

# Create a dataset called test_dataset
def getOrCreate_dataset(dataset_name :str, project_id = bigquery_client.project) -> bigquery.dataset.Dataset:

    '''
    Get dataset. If the dataset does not exists, create it.
    
    Args:
        - dataset_name(str) = Name of the new/existing data set.
        - project_id(str) = project id(default = The project id of the bigquery_client object)

    Returns:
        - dataset(google.cloud.bigquery.dataset.Dataset) = Google BigQuery Dataset
    '''

    print('Fetching Dataset...')

    try:
        # get and return dataset if exist
        dataset = bigquery_client.get_dataset(dataset_name)
        print('Done')
        print(dataset.self_link)
        return dataset

    except Exception as e:
        # If not, create and return dataset
        if e.code == 404:
            print('Dataset does not exist. Creating a new one.')
            bigquery_client.create_dataset(dataset_name)
            dataset = bigquery_client.get_dataset(dataset_name)
            print('Done')
            print(dataset.self_link)
            return dataset
        else:
            print(e)
 

def getOrCreate_table(dataset_name:str, table_name:str) -> bigquery.table.Table:


    '''
    Create a table. If the table already exists, return it.
    
    Args:
        - table_name(str) = Name of the new/existing table.
        - dataset_name(str) = Name of the new/existing data set.
        - project_id(str) = project id(default = The project id of the bigquery_client object)

    Returns:
        - table(google.cloud.bigquery.table.Table) = Google BigQuery table
    '''

    # Grab prerequisites for creating a table
    dataset = getOrCreate_dataset(dataset_name)
    project = dataset.project
    dataset = dataset.dataset_id
    table_id = project + '.' + dataset + '.' + table_name

    print('\nFetching Table...')

    try:
        # Get table if exists
        table = bigquery_client.get_table(table_id)
        print('Done')
        print(table.self_link)
    except Exception as e:

        # If not, create and get table
        if e.code == 404:
            print('Table does not exist. Creating a new one.')
            bigquery_client.create_table(table_id)
            table = bigquery_client.get_table(table_id)
            print(table.self_link)
        else:
            print(e)
    finally:
        return table


