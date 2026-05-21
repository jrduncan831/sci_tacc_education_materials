from agent_codebase.tools import *
import ollama

# Sends the prompt to the LLM and returns the message response string
def llm_prompt(prompt: str, 
               system_message: str="You are a helpful AI Agent named Kiwi.", 
               seed: int=-1, 
               model: str="llama3.2") -> str:

        # generate a text response by sending our prompt to the ollama server 
        response = ollama.chat(
            model="llama3.2", 
            options={"seed":seed}, 
            messages=[
            {"role": "system", "content": system_message},{"role": "user", "content": prompt}],
        )
        return response['message']['content']

# Sends the prompt to the LLM and returns the message response string
def llm_prompt_tool(prompt: str, 
               tools: list,
               system_message: str="You are a helpful AI Agent named Kiwi.", 
               seed: int=-1, 
               model: str="llama3.2") -> str:

        response = ollama.chat(
            model="llama3.2", 
            options={"seed":seed}, 
            messages=[
            {"role": "system", "content": system_message},{"role": "user", "content": prompt}],
            tools=tools
        )

        # pull out the tool call dictionary from the response object
        tools_calls = response['message']['tool_calls']
        print(f"Tool call:\n{tools_calls[0]}\n") 
    
        # get the name and input arguments of the tool
        tool_name = tools_calls[0]['function']['name']
        arguments = tools_calls[0]['function']['arguments']

        # execute the tool
        result = globals()[tool_name](**arguments)
    
        return result

# formats the prompt into a template string that asks for list in response
def create_list_template(prompt: str) -> str:

    list_schema = """```json
{
"list_description": "Describe list contents here",
"content":[
{"name": "Name of list item 1", "description": "Description of list item 1"},
{"name": "Name of list item 2", "description": "Description of list item 2"},
...
]
}
```"""
    
    output_prompt = f"""The following prompt requires you to respond with a list.

*Prompt:* {prompt}

*List format schema:*
{list_schema}

The "..." indicates that the list can be as many items long as you require.

Respond with the list that address the prompt using the above formatting schema."""
    
    return output_prompt

import json
# Given a prompt, ask the LLM to make a list in response, retry until it responds in a good format
# Will respond with a dictionary with a list 'description' and 'content' which is an array of dicts
# with 'name' and 'description' of each list item.  An empty dict is returned if it fails to make
# a well formatted list
def llm_create_list(prompt: str, model="llama3.2") -> dict:

    # Desired system message
    system_message = "You are a helpful AI Agent named Kiwi."

    # Embed our prompt into the pick_option_template
    prompt = create_list_template(prompt)
    
    # # For testing, print templated prompt
    # print(f"Templated Prompt:\n{prompt}")

    # set the choice to no choice selected
    generated_list = {}
    
    # We'll try 10 times to get a valid choice
    for seed in range(1, 10):

        # # For testing, print current seed value
        # print(f"Querying model with random seed vaule {seed}...\n")

        # print progress
        status = f"trying to create list [{seed}/10] times..." 
        print(status,end='')
        
        # Send our pick tools prompt to the model
        response_text = llm_prompt(prompt, system_message, model=model)
        
        # print(response_text)
        # try to convert the LLM response to an int
        try:

            # cleaned_string = response_text.strip("``````").strip()
            # cleaned_string = cleaned_string.removeprefix("json")

            # find the json data in the response which should start with ```json and end with ```
            start_index = response_text.find("```json")+7
            end_index = response_text.find("```", start_index)

            # pull out just the json data
            cleaned_list_string = response_text[start_index:end_index]

            # load into a json object
            generated_list = json.loads(cleaned_list_string)
            
            # Display success message
            print(f"success! :D\n")

            # print(f"Cleaned LLM Response:\n{cleaned_list_string}")
            break
            
        except Exception as e:
            print(f"An error occurred parsing the generated list json:\n {e}")
            # For testing, print response
            # print(f"Raw LLM Response:\n{response_text}\n")
            # print(f"\n\n Cleaned LLM Response:\n{response_text.strip('```json').split('```')[0].strip()}")
        
    return generated_list

# Takes a list of dictionary objects with the 'name' and 'description'
# and builds a string with a numbered list. The none_option flag 
# inserts an "Option 0: None of these" into the choices
def build_option_list(options: list,none_option: bool=False) -> str:

    # add an option to selection none
    if none_option:
        output_string = "Option 0: None of these\n\n"
        i_offset = 1
    else:
        output_string = ""
        i_offset = 0

    # build the rest of the options from the input options list
    output_string += "\n\n".join(f"Option {i+i_offset}: {option.get('name')}, Description: {option.get('description')}" for i, option in enumerate(options))

    return output_string

# formats the prompt and available options to choose from into a template string
def pick_option_template(prompt: str, options: str) -> str:

    output_prompt = f"""Choose which option is best suited to address the prompt.

*Prompt:* {prompt}

*Options:*\n{options}

Respond with a single number."""
    
    return output_prompt

# Asks the LLM which option (array of dictionaries with 'name' and 'description') to use to answer the prompt
# returns the LLM's choice as a single integer
# if non_option = True, choice = 0 means "None of these", otherwise 0 means the first option you provided.
# A choice = -1 means the model could not make a valid choice (it will try 100 seeds by default before giving up)
def llm_pick_option(prompt: str, options: list, none_option: bool=False, show_prompt: bool=False) -> int:

    # Desired system message
    system_message = "You are a helpful AI Agent named Kiwi."

    # Convert to a string that contains the list of options
    options_list = build_option_list(options, none_option)

    # For testing, print out the options string we will send
    # print("Options List:\n\n" + options_list + "\n")

    # number of options to chose from
    if none_option:
        num_options = len(options)+1
    else:
         num_options = len(options)

    # Embed our prompt into the pick_option_template
    prompt = pick_option_template(prompt, options_list)
    
    # print out prompt if we want it
    if show_prompt:
        print(f"Templated Prompt:\n{prompt}")

    # set the choice to no choice selected
    choice = -1
    
    # We'll try 100 times to get a valid choice
    for seed in range(1, 101):

        # For testing, print current seed value
        # print(f"Querying model with random seed vaule {seed}...\n")

        # Send our pick tools prompt to the model
        response_text = llm_prompt(prompt, system_message)
        
        # try to convert the LLM response to an int
        try:
            choice = int(response_text)

            if choice >=0 and choice < num_options:
                # For testing, displauy chosen option
                # print(f"Chose option {choice}!")
                break
            
        except:
            pass
            # For testing, display that it failed to make a choice
            # print('Failed to convert response to int :/')
        
    return choice