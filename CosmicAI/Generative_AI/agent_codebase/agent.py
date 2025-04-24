from agent_codebase.llm_functions import llm_create_list, llm_prompt_tool, llm_prompt

def run_agent(user_prompt, tools, model="llama3.2"):

    print(f"User Prompt: {user_prompt}\n")
    functions = [f["function"]["name"] for f in tools]
    nl = "\n"
    print(f"Tools the LLM has access to:\n{nl.join([f'{f}' for f in functions])}\n")

    # Format our goal and tool descriptions into a step-by-step (sbs) plan template
    sbs_plan_template = f"""\nGoal: {user_prompt}.

    Tools you can use:
    {tools}

    Break up your goal into subgoals that utilize these tools. Ensure each step involves only a single use of a tool! If you want to use a tool twice, break it up into two steps."""

    print('Generating step by step plan...')

    # create the step by step plan template
    sbs_plan = llm_create_list(sbs_plan_template, model=model)

    # string to hold numbered list with step by step plan names
    sbs_plan_names = ""

    # Print plan to console or quit if we failed to make one
    if bool(sbs_plan):
        # pull out list names
        step_names = [item['name'] for item in sbs_plan['content']]

        # print list to console for viewing
        sbs_plan_names = f"{nl.join([f'{i+1}. {name}' for i, name in enumerate(step_names)])}\n"
        print("Generated Step by Step plan:\n" + sbs_plan_names)
    else:
        print(f"No plan generated :(")
        sys.exit()

    # string array to hold responses from each step
    subgoal_results = []

    # Run tool calls on each step
    for i, step in enumerate(sbs_plan['content']):

        step_title = (
            "===================================================================\n"
            f"Executing step [{i+1}/{len(sbs_plan['content'])}]"
            f" {sbs_plan['content'][i]['name']}\n"
        )
        print(step_title)
        
        # formate subgoal prompt template
        subgoal_prompt_template = f"""Goal: {user_prompt}.

    Step by step plan:
    {sbs_plan_names}
    """

        # print(subgoal_prompt_template)
        
        if i>0:
            # build formated string of previous sub goal results      
            past_results = "\n".join([f"""Step {step_id+1}: {sbs_plan['content'][step_id]['name']}
        {subgoal_results[step_id]}""" for step_id in range(0,i)])

            # add to our prompt template
            subgoal_prompt_template+= "Results of previous steps:\n\n" + past_results + "\n\n"

        # add the final instruction to our prompt template
        subgoal_prompt_template+=f"""Complete the current step: {step['name']}
    {step['description']}
    """   

        # print(f"Subgoal Template {i+1}:\n{subgoal_prompt_template}")
        
        # get new subgoal results, attempt 10 times if we fail
        for attempt in range(10):
            try:
                tool_results = llm_prompt_tool(subgoal_prompt_template,tools, model=model)
                print(f'tool_results: {tool_results}')
                subgoal_results.append(tool_results)
                break
            except Exception as e:
                print(f'Error with tool call: {e}')
                if attempt == 9:
                    subgoal_results.append("Failed to execute tool.")

        # print(f"\nResponse to step {i+1}:\n{subgoal_results[i]}\n\n")

    # write out the results of each subgoal
    final_subgoal_results = "\n".join([f"""Step {step_id+1}: {sbs_plan['content'][step_id]['name']}
        {subgoal_results[step_id]}""" for step_id in range(len(subgoal_results))])

    # formate final prompt template
    final_prompt_template = f"""Goal: {user_prompt}.

    Step by step plan:
    {sbs_plan_names}

    Results from each step:
    {final_subgoal_results}

    Complete the goal. Respond only with the answer.
    """

    # perform final prompt call
    response = llm_prompt(final_prompt_template, model=model)
    print("\n====================== Final Prompt Template ======================\n")
    print(final_prompt_template)
    print("===================================================================\n\n")

    print(f"Final Response:\n{response}")