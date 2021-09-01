import datarobot as dr
from datarobot.errors import ClientError

import pickle

import logging
from warnings import warn
from warnings import catch_warnings

LOGGER = logging.getLogger(__name__)

def find_project(project: str) -> dr.models.project.Project:
    """Uses the DataRobot api to find a current project.

    Uses dr.Project.get() and dr.Project.list() to test if 'project' is either an id
    or possibly a name of a project in DataRobot, then returns the project found.

    ## Parameters:
    ----------
    project: str
        Either a project id or a search term for project name

    ## Returns: 
    ----------
    dr.models.project.Project
        A Datarobot project object that is either the project with the id provided, or the 
        first/only project returned by searching by project name. Returns None if the list is 
        empty.
    """
    project = str(project)
    return_project = None
    try:
        return_project = dr.Project.get(project_id=project)
        return return_project
    except ClientError:
        project_list = dr.Project.list(search_params={'project_name': project})
        if len(project_list) == 0:
            warn(f"No projects found with id or search for \'{project}\'")
            return None
        elif len(project_list) == 1:
            return project_list[0]
        else:
            warn(f"Returning the first of multiple projects with \'{project}\': {project_list=}")
            return project_list[0]


def initialize_dr_api(token_key, file_path: str = 'data/dr-tokens.pkl', endpoint: str = 'https://app.datarobot.com/api/v2'):
        """Initializes the DataRobot API with a pickled dictionary created by the user.

        <mark>WARNING</mark>: It is advised that the user keeps the pickled dictionary in an ignored 
        directory if using GitHub (put the file in the .gitignore)

        Accesses a file that should be a pickled dictionary. This dictionary has the API token
        as the value to the provided token_key. Ex: {token_key: 'API_TOKEN'}

        ## Parameters
        ----------
        token_key: str | int | etc...
            The API token's key in the pickled dictionary located in file_path

        file_path: str, optional (default = 'data/dr-tokens.pkl')
            Path to the pickled dictionary containing the API token

        endpoint: str, optional (default = 'https://app.datarobot.com/api/v2')
            The endpoint is usually the URL you would use to log into the DataRobot Web User Interface

        """
        # load pickled dictionary and initialize api, catching FileNotFound, KeyError, and failed authentication warning
        try:
            datarobot_tokens = pickle.load(open(file_path, 'rb'))
            with catch_warnings(record=True) as w: # appends warning to w if warning occurs
                dr.Client(endpoint=endpoint, token=datarobot_tokens[token_key])
                if not not w: # check to see if w is not None or empty (has a warning)
                    raise Exception(w[0].message)
                else:
                    pass
        except FileNotFoundError:
            raise FileNotFoundError(f'The file {file_path} does not exist.') # TODO: Make a tutorial on how to create the pickled dictionary with api tokens and link here
        except KeyError:
            raise KeyError(f'\'{token_key}\' is not in the dictionary at \'{file_path}\'')
        
        print(f'DataRobot API initiated with endpoint \'{endpoint}\'')