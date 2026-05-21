import inspect
import json
import re
def generate_function_description(func):
    func_name = func.__name__
    docstring = func.__doc__

    # Get function signature
    sig = inspect.signature(func)
    params = sig.parameters

    # Create the properties for parameters
    properties = {}
    required = []

    # Process the docstring to extract argument descriptions
    arg_descriptions = {}
    if docstring:
        # remove leading/trailing whitespace or leading empty lines and split into lines
        docstring = re.sub(r'^\s*|\s*$', '', docstring, flags=re.MULTILINE)
        lines = docstring.split('\n')
        current_arg = None
        for line in lines:
            line = line.strip()
            if line:
                if ':' in line:
                    # strip leading/trailing whitespace and split into two parts
                    line = re.sub(r'^\s*|\s*$', '', line)
                    parts = line.split(':', 1)
                    if parts[0] in params:
                        current_arg = parts[0]
                        arg_descriptions[current_arg] = parts[1].strip()
                elif current_arg:
                    arg_descriptions[current_arg] += ' ' + line.strip()

    for param_name, param in params.items():
        param_type = 'string'  # Default type; adjust as needed based on annotations
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation.__name__.lower()

        param_description = arg_descriptions.get(param_name, f'The name of the {param_name}')

        properties[param_name] = {
            'type': param_type,
            'description': param_description,
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Create the JSON object
    function_description = {
        'type': 'function',
        'function': {
            'name': func_name,
            'description': docstring.split('\n')[0] if docstring else f'Function {func_name}',
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            },
        },
    }

    return function_description

import requests
import time
from duckduckgo_search import DDGS


def get_duckduckgo_result(query: str) -> str:
    """
    Search for information online for the given query using DuckDuckGo.
    query: The search query to send to DuckDuckGo.
    """
    results = DDGS().text(query, 
                          region='wt-wt', 
                          safesearch='off', 
                          timelimit='y', 
                          max_results=10)
    
    return results[0]['body']

def do_math(a:int, op:str, b:int)->list:
    """
    Performs math on the inputs
    a: The first operand
    op: The operation to perform (one of '+', '-', '*', '/', '<')
    b: The second operand
    """
    res = "Nan"
    if op == "+":
        res = str(int(a) + int(b))
    elif op == "-":
        res = str(int(a) - int(b))
    elif op == "*":
        res = str(int(a) * int(b))
    elif op == "/":
        if int(b) != 0:
            res = str(int(a) / int(b))
    elif op == '<':
        res = str(a < b)
    return res

def get_current_time() -> str:
    """Get the current time"""
    current_time = time.strftime("%H:%M:%S")
    return f"The current time is {current_time}"

def get_current_weather(city:str) -> str:
    """Get the current weather for a city
    Args:
        city: The city to get the weather for.  Respond only with the city name, no state designation.
    """
    base_url = f"http://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"The current temperature in {city} is: {data['current_condition'][0]['temp_C']}°C"

