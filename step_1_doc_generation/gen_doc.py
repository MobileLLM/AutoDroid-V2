import argparse
import os
import tools as tools
from describe_interactions import describe
from build_xpath import ScreenSkeletonBuilder, XPathBuilder
from build_dependency import DependencyGraph
from post_process_doc import post_process
from extract_additional_elements import extract_additional_elements
from pathlib import Path
import copy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from dotenv import load_dotenv

load_dotenv()

MAX_RETRY = 10

def parse_args():
    parser = argparse.ArgumentParser(description="This script generates solutions.")
    parser.add_argument('-m', '--model', default="gpt-4o", help='Model name')
    parser.add_argument('-t', '--timestamp', default="1025", help='Timestamp, using to distinguish the file')
    parser.add_argument('-o', '--output_path', default='output/llama_touch/docs/settings_1025', help='Path to output file')
    parser.add_argument('-d', '--annotation_dir', default='data/llama_touch/explore_data/settings', help='Path to annotation directory')
    parser.add_argument('-a', '--app_name', default="Discord")
    parser.add_argument('-i', '--include_image', action='store_true', default=True, help='Include images in the output')

    args = parser.parse_args()

    return args

def split_mismatched_screen(app_name, screen_data, tag_state, temperature):
    prefix = f"You are an app tester and you are testing a {app_name} app on a smartphone. You currently collect a series of UIs, the name of which are {screen_data['api_name']}. However, the layout of these UIs are not similar, therefore, they are not the same type of UIs. The UIs (described in HTML) are as follows:\n\n"
    id_to_tag_mapping = {}
    all_states_desc = ''
    for tagid, tag in enumerate(screen_data['tags']):
        all_states_desc += f"UI{tagid}: \n{tag_state[tag]}\n\n"
        id_to_tag_mapping[f"UI{tagid}"] = tag
    instruction = f"""You should classify the above UIs into different categories based on their layout. The output format should be a json dict: 
```json
{{
    "1": {{
        "uis": ["UI0", "UI1", ...],
        "description": "description of the layout"
        }},
    "2": {{
        "uis": ["UI2", "UI3", ...],
        "description": "description of the layout"
        }},
   ...
}}
```
You should note that: 
  - DON'T classify the UIs based on their content (like date, time, name, text, descriptions, etc.), only based on their layout, classify the UIs if their layouts are very different. 
  - DON'T classify into too many categories. DON'T use more than 3 categories.
  - **Output only a valid JSON response in the form of the above mentioned document. Enclose every property name and value in double quotes. It must be a valid JSON response!**
  - Note that all your 'uis' should include all the UIs from UI0 to UI{len(screen_data['tags'])-1} in the output.
"""
    prompt = prefix + all_states_desc + instruction
    tools.write_txt_file('temp/prompts/split_screen_prompt.txt', prompt)
    answer = tools.debug_query_gptv2(prompt, 'gpt-4o', temperature=temperature)
    print(answer)
    answer = tools.convert_gpt_answer_to_json(answer, 'gpt-4o')

    new_screen_datas = {}
    splitted_ui_ids = []
    for category, cate_data in answer.items():
        tags = [id_to_tag_mapping[ui] for ui in cate_data['uis']]
        splitted_ui_ids += cate_data['uis']
        new_screen_data = copy.deepcopy(screen_data)
        new_screen_data['tags'] = tags
        new_screen_datas[f"{screen_data['api_name']}_v{category}"] = new_screen_data

    all_uis_matched = True
    for ui_ids in id_to_tag_mapping.keys():
        if ui_ids not in splitted_ui_ids:
            all_uis_matched = False
            break

    return new_screen_datas, all_uis_matched

def split_mismatched_screen_by_rule(screen_data, tag_state):
    # split by structure string
    layouts, layoutid_to_tags = [], {}
    for tagid, tag in enumerate(screen_data['tags']):
        state = tag_state[tag]
        layout = tools.clean_repeated_siblings(tools.clean_attributes(state))
        if layout not in layouts:
            layouts.append(layout)
            layoutid_to_tags[layouts.index(layout)] = [tag]
        else:
            layoutid_to_tags[layouts.index(layout)].append(tag)
    
    new_screen_datas = {}
    for layoutid, tags in layoutid_to_tags.items():
        new_screen_data = copy.deepcopy(screen_data)
        new_screen_data['tags'] = tags
        new_screen_datas[f"{screen_data['api_name']}_v{layoutid}"] = new_screen_data

    return new_screen_datas, True

def build_tag_former_ui_action_lookup_table(step_history, raw_log):

    former_ui_for_each_tag = {}
    former_steps = {}
    for record_id, record in enumerate(raw_log['records']):
        
        if str(record_id) not in step_history.keys() or str(record_id-1) not in step_history.keys():
            if str(record_id) in step_history.keys():
                current_ui = step_history[str(record_id)]['ui']
                # maybe we iterate to the end and does not find the target tag, this case, the tag is some UI we continued, then we view it as the first UI
                former_ui_for_each_tag[record['tag']] = {'ui': current_ui, 'action': 'open_app'}
                former_steps[record_id] = {'ui': current_ui, 'action': 'open_app'}
            
            continue

        
        former_ui_id = list(former_steps.keys())[-1]
        while former_ui_id > 0:
            if former_ui_id in former_steps.keys() and former_steps[former_ui_id]['ui'] != current_ui:
                break
            former_ui_id -= 1
        if former_ui_id == 0:
            former_ui_for_each_tag[record['tag']] = {'ui': 'None, this is the first UI', 'action': 'open_app'}
        else: 
            former_ui_for_each_tag[record['tag']] = {'ui': former_steps[former_ui_id]['ui'], 'action': former_steps[former_ui_id]['action']}
        former_steps[record_id] = {'ui': step_history[str(record_id)]['ui'], 'action': step_history[str(record_id)]['action']}
    return former_ui_for_each_tag

def build_tag_uiname_lookup_table(step_history, raw_log):
    tag_uiname_lookup_table = {}
    for record_id, record in enumerate(raw_log['records']):
        if str(record_id) not in step_history.keys():
            continue
        tag_uiname_lookup_table[record['tag']] = step_history[str(record_id)]['ui']
    return tag_uiname_lookup_table

def describe_each_splitted_screen(app_name, screen_data, tag_state, raw_log, step_history, all_screen_data, old_api_names):
    prefix = f"""You are an app tester and you are testing a {app_name} app on a smartphone. You currently collect a series of UIs, you should give an api name to these UIs for further reference. You should follow the following instructions: 
## Input: 
    * HTML description of the UIs needed to summarize into an api name. 
    * The former UI api name of each UI. 
    * the action in the former UI api name that leads to the the transfer from the former UI to the current UI. 

## General Rule: Avoid Specific Content
    * The description and the api name must avoid specific content and details. Use general terms to describe elements and interactions.
    * Examples:
        - Instead of "a screen showing the current date 'Sun, Oct 15'", use "a screen showing the current date"
        - Instead of "a button labeled 'Sun, Oct 15'", use "a date button".
        - Instead of "the comment is updated to 'this picture is great'", use "the comment is updated according to the user input".
        - Instead of "the comment is updated to 'this picture is great'", use "the comment is updated according to the user input".
    * The API name must be concise and informative regarding the screen's purpose, based on the functionality of the current UI. 
    * Please name it by focusing on its function, which can be inferred from the current UI, the former UI api_name, and the action that leads to the current UI. 

The UIs (described in HTML) are as follows:\n\n"""
    former_ui_lookup_table = build_tag_former_ui_action_lookup_table(step_history, raw_log)
    current_ui_lookup_table = build_tag_uiname_lookup_table(step_history, raw_log)
    all_states_desc = ''
    
    for id, tag in enumerate(screen_data['tags']):
        if tag not in former_ui_lookup_table.keys() or tag not in current_ui_lookup_table.keys():
            print('tag not in former_ui_lookup_table or current_ui_lookup_table')
            continue
        former_ui_name = former_ui_lookup_table[tag]['ui']
        current_ui_name = current_ui_lookup_table[tag]
        if former_ui_name == 'None, this is the first UI':
            former_desc = 'None, this is the first UI'
        else:
            former_desc = all_screen_data[former_ui_name]['description']
        all_states_desc += f"### UI {id}: {all_screen_data[current_ui_name]['description']}\n"
        all_states_desc += f"HTML of UI {id}: \n{tag_state[tag]}\n"
        all_states_desc += f"Former UI of UI {id}: {former_desc}\n"
        all_states_desc += f"Action in the former UI: {former_ui_lookup_table[tag]['action']}\n\n"
        
    instruction = f"""You should summarize the above UIs (namely {', '.join([f'UI {tagid}' for tagid in range(len(screen_data['tags']))])}) into a concise and informative API name. If the UIs are a little different, use 'or' to separate the different parts, such as "select_event_start_or_end_date". The output format should be a json dict:
```json
{{
    "api_name": "<api_name>", 
    "description": "<a description of the function of this type of UIs>" // you should be informative, both include a description of the UI and its functionality. For example: "The screen allows user to customize a reminder to occur before the scheduled time of the event being edited. It displays a dialog for setting a custom reminder interval. The dialog includes an input field for the interval value and radio buttons for selecting the unit of time (minutes, hours, days). Below the input field, there are 'Cancel' and 'OK' buttons."
}}
```

Note that: 
- The api_name must be related to the functionality of the UIs, which could be inferred from the former UI, the action that leads to the current UI (**but IGNORE the former UI and the action if the action is like 'go back'**). **DON'T** directly use the resource_id in the current UI as the api_name, but be careful about its functionality. 
- **DON'T use any of these api_names because they have been assigned to other UIs: {', '.join(old_api_names)}**
- **DON'T use any of the former api_names as the new api_name in your response!**
- Output **only** a valid JSON response in the form of the above mentioned document. Enclose every property name and value in double quotes. It must be a valid JSON response!
"""
    prompt = prefix + all_states_desc + instruction
    tools.write_txt_file('temp/prompts/describe_split_screen_prompt.txt', prompt)
    answer = tools.debug_query_gptv2(prompt, 'gpt-4o')
    print(answer)
    answer = tools.convert_gpt_answer_to_json(answer, 'gpt-4o')
    screen_data['api_name'] = answer['api_name']
    screen_data['description'] = answer['description']
    return screen_data

def check_mismatched_uis(app_name, descriptions_output_file, annotation_log, screens, tag_state, output_path, timestamp):
    if os.path.exists(f"{output_path}/descriptions_{timestamp}_after_split.json"):
        return
    skeleton_builder = ScreenSkeletonBuilder(f"{descriptions_output_file}.json", f"{descriptions_output_file}_states.json")
    skeleton_builder.extract_skeleton_of_all_screens()
    mismatched_screens = skeleton_builder.mismatched_screen_names
    print('mismatched_screens:', mismatched_screens)

    new_screens = {}
    raw_log = tools.load_yaml_file(annotation_log)
    step_history = tools.load_json_file(f"{descriptions_output_file}_step_history.json")
    for screen, screen_data in screens.items():
        if screen in mismatched_screens:
            # new_screen_datas, all_uis_matched = split_mismatched_screen(app_name, screen_data, tag_state, temperature=0.2)
            new_screen_datas, all_uis_matched = split_mismatched_screen_by_rule(screen_data, tag_state)

            retry_times, randomness = 0, 0.3
            while not all_uis_matched and retry_times < MAX_RETRY:
                new_screen_datas, all_uis_matched = split_mismatched_screen(app_name, screen_data, tag_state, temperature=randomness)
                retry_times += 1
                randomness += 0.05

            forbidden_screen_names = [screen]
            logging.info(f"split screen {screen} into {new_screen_datas.keys()}")            
            for new_screen, new_screen_data in new_screen_datas.items():
                new_screen_data = describe_each_splitted_screen(app_name, new_screen_data, tag_state, raw_log, step_history, screens, forbidden_screen_names)
                if new_screen_data["api_name"] not in new_screens.keys():
                    new_screens[new_screen_data['api_name']] = new_screen_data
                else:
                    new_screens[new_screen_data['api_name']]['tags'] += new_screen_data['tags']
                    new_screens[new_screen_data['api_name']]['interactions'] += new_screen_data['interactions']
                    
                forbidden_screen_names.append(new_screen_data['api_name'])
        else:
            new_screens[screen] = screen_data

    # save the new screens to file
    tools.dump_json_file(f"{output_path}/descriptions_{timestamp}_after_split.json", new_screens)

if __name__ == '__main__':
    
    # timestamp = "07100300"
    # output_path = "data/doc/broccoli"
    # model = "gpt-4o"
    # annotation_dir = "output/broccoli"
    args = parse_args()

    timestamp = args.timestamp
    output_path = args.output_path
    model = args.model
    annotation_dir = args.annotation_dir

    dir_path = Path(output_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    annotation_log = os.path.join(annotation_dir, f'log.yaml')
    annotation_states = os.path.join(annotation_dir, 'states')

    descriptions_output_file = os.path.join(output_path, f'descriptions_{timestamp}')
    extractions_output_file = os.path.join(output_path, f'extraction')
    doc_output_file = os.path.join(output_path, f'doc.json')

    describe_prompts_answers_path = f'doc_generation/temp/describe_prompts_answers_{timestamp}.json'

    '''screens: {
       screen_api_name(title): 
       {
            api_name, 
            description, 
            tags: [a list of tag strings], 
            interactions: [a list of dict: {'action', 'description', 'effect', 'api_name', 'current_screen', 'next_screen', 'element', 'element_type', 'input', 'action_type', 'state_tag'}]
       }
    }; 
    tag_state: {state_tag: <state HTML description>}; '''
    # screens, tag_state = describev2(annotation_log,annotation_states, descriptions_output_file, model, describe_prompts_answers_path)
    screens, tag_state = describe(annotation_log,annotation_states, descriptions_output_file, model, describe_prompts_answers_path, include_image=args.include_image)

    check_mismatched_uis(args.app_name, descriptions_output_file, annotation_log, screens, tag_state, output_path, timestamp)
    
    extract_additional_elements(
        annotation_log_path=annotation_log, 
        annotation_states_path=annotation_states,
        descriptions_file_path=f'{descriptions_output_file}_after_split.json', 
        output_file_name=extractions_output_file, 
        prompt_answer_path=f'{extractions_output_file}_prompt_answer.json',
        model='gpt-4o', 
        include_image=args.include_image
    )
    # for droidtask dataset, we do not include image, and the description and text are used for xpath generation
    use_desc_and_text = not args.include_image
    xpath_builder = XPathBuilder(f'{descriptions_output_file}_after_split.json', f'{descriptions_output_file}_states.json', f"{extractions_output_file}_screen_elements.json", use_desc=use_desc_and_text, use_text=use_desc_and_text)
    xpath_builder.save_xpath_and_skeleton_to_file(doc_output_file)

    dep_graph = DependencyGraph(annotation_log, doc_output_file, f"{extractions_output_file}_tag_elements.json", f"{descriptions_output_file}_step_history.json")
    dep_graph.show_graph(path=output_path)
    dep_graph.get_all_elements_paths()
    
    post_process(doc_output_file)
