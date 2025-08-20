import json
import os
import tools as tools
import copy
import ast
import parallel_query as mtp
from itertools import combinations
import re
from argparse import ArgumentParser
from parallel_query import MultiProcessingQuery


def parseargs():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['doc_format', 'task', 'solution', 'pre_task', 'pre_task_solution', 'hard_task',
                                 'hard_task_solution'],
                        help='doc_format: format the api paths for raw api document; task: generate tasks for apps; solution: generate solutions for apps; pre_task: generate pre_task for current task; hard_task: generate more complex tasks.')
    parser.add_argument('--query', action='store_true', help='whether to call api to query GPT')
    return parser.parse_args()


class APIPathSolver:
    def __init__(self, apis_folder_path):
        self.apis_folder_path = apis_folder_path
        self.all_api_paths = []
        self.unique_apis = self.solve_all_api_paths(apis_folder_path)

    def load_data(self, path):
        # eles_data = json.load(open(os.path.join(path, 'ele.json')))
        apis_data = json.load(open(os.path.join(path, 'apis.json')))
        # tree_data = json.load(open(os.path.join(path, 'tree.json')))

        apis = {}
        dep_in = set()  # roots of the dependency tree(forrest)
        dpe_edge = {}
        for _, v in apis_data.items():
            for e in v:
                if e["name"] == "":
                    continue
                name = e["name"]
                apis[name] = apis.get(name, e)
                dep = apis[name]["dependency"]
                if len(dep) == 0:
                    dep_in.add(name)
                for d in dep:
                    if d == '':
                        dep_in.add(name)
                        continue
                    p_name = d[0:-1].split('(')[-1]
                    if d.startswith('window'):
                        dep_in.add(name)
                        continue
                    dpe_edge[p_name] = dpe_edge.get(p_name, [])
                    dpe_edge[p_name].append(name)

        dep_tree = {
            "name": "root",
            "children": [{
                "name": n,
                "children": [],
            } for n in dep_in]
        }

        queue: list = dep_tree['children'].copy()
        while len(queue) > 0:
            cur = queue.pop(0)
            edge = dpe_edge.get(cur["name"], [])
            for n in edge:
                tmp = {"name": n, "children": []}
                cur['children'].append(tmp)
                queue.append(tmp)

        return apis, dep_tree

    def dfs(self, node, p: list):
        p.append(node['name'])
        if len(node['children']) == 0:
            self.all_api_paths.append(p)
            return
        for n in node['children']:
            self.dfs(n, p.copy())

    def solve_all_api_paths(self, apis_folder_path):
        apis, dep_tree = self.load_data(apis_folder_path)
        self.dfs(dep_tree, [])
        return apis

    def get_path_by_api_name(self, api_name):
        # search the dependency path of the given api_name in the given paths
        all_paths_to_api = []
        for path in self.all_api_paths:
            if api_name in path:
                api_name_index = path.index(api_name)
                path_to_api = path[:api_name_index + 1]
                if path_to_api not in all_paths_to_api:
                    all_paths_to_api.append(path_to_api)
        return all_paths_to_api

    def _search_dependency_in_original_apis_json(self, apis, api_name):
        for time_tag, state_data in apis.items():
            for ele in state_data:
                if ele['name'] == api_name:
                    return ele['dependency']
        return []

    def _get_api_action_type(self, dependency):
        if 'tap(' in dependency.lower() or 'touch(' in dependency.lower():
            return 'touch'
        elif 'long_touch' in dependency.lower() or 'long_press' in dependency.lower() or 'long_tap' in dependency.lower():
            return 'long_touch'
        elif 'scroll up' in dependency.lower():
            return 'scroll up'
        elif 'scroll down' in dependency.lower():
            return 'scroll down'
        elif 'input(' in dependency.lower() or 'set_text(' in dependency.lower():
            return 'set_text'
        elif 'select(' in dependency.lower():
            return 'select'
        elif 'match(' in dependency.lower():
            return 'match'
        else:
            return 'unknown'

    def get_path_with_action_type_by_api_name(self, api_name):
        '''
        search the dependency path of the given api_name in the given paths
        '''
        apis_data = json.load(open(os.path.join(self.apis_folder_path, 'apis.json')))

        all_paths_to_api = []
        for path in self.all_api_paths:
            if api_name in path:
                api_name_index = path.index(api_name)
                path_to_api = path[:api_name_index + 1]
                if path_to_api not in all_paths_to_api:
                    # add action type for each api in the path
                    for i in range(len(path_to_api) - 1):
                        if path_to_api[i] == 'root':
                            continue
                        current_dependency = path_to_api[i]
                        # search the dependency data of the next api, match the current api, then get the action type
                        original_dependency_data = self._search_dependency_in_original_apis_json(apis_data,
                                                                                                 path_to_api[i + 1])

                        for d in original_dependency_data:
                            if current_dependency in d:
                                current_dependency_action_type = self._get_api_action_type(d)

                        path_to_api[i] = {'name': path_to_api[i], 'action_type': current_dependency_action_type}
                    if path_to_api not in all_paths_to_api:
                        all_paths_to_api.append(path_to_api)
        return all_paths_to_api

    def get_path_for_all_apis(self):
        api_paths = {}
        for api_name in self.unique_apis:
            api_paths[api_name] = self.get_path_by_api_name(api_name)
        return api_paths

    def add_action_type_for_dependencies(self):
        api_paths = {}
        for api_name in self.unique_apis:
            api_paths[api_name] = self.get_path_with_action_type_by_api_name(api_name)
        return api_paths


def get_semantic_dependencies(dep_list):
    if len(dep_list) == 0:
        return 'No dependency, this UI element is in the main screen of the app'
    semantic_dependencies = ''
    for dependency_id, dependency in enumerate(dep_list):
        if 'window(' in dependency:
            element_window = dependency.split('(')[-1][:-1]
            semantic_dependency = f'this UI element could be reached in the {element_window} screen of the app'
        elif dependency == '':
            semantic_dependency = 'this UI element could be interacted in the main screen'
        else:
            semantic_dependency = f'this UI element could be interacted after {dependency}'
        semantic_dependencies += f'{semantic_dependency}'
        if dependency_id != len(dep_list) - 1:
            semantic_dependencies += ' or '
    return semantic_dependencies


class TaskGenerator:
    def __init__(self, apis_path, api_tree_paths_path, app_name):
        self.name_to_desc = {}
        self.name_to_func = {}
        self.dependencies = {}
        self.all_prompts = {}

        self.apis_path = apis_path
        self.api_tree_paths_path = api_tree_paths_path
        self.app_name = app_name

    def init_construct_map(self):

        with open(self.apis_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for timestamp, elements in data.items():
            # 每个时间戳对应的 element
            for element in elements:
                name = element['name']
                desc = element['desc']
                func = element['func']

                self.name_to_desc[name] = desc
                self.name_to_func[name] = func

    def extract_dependencies(self, data):
        cnt = 0
        for key, value in data.items():
            if isinstance(value, list):
                tmp = []
                for item in value:
                    if isinstance(item, list) and len(item) >= 3:
                        father_node = item[-2]
                        action = father_node['action_type'] + '(' + father_node['name'] + ')'
                        tmp.append(action)
                if tmp:
                    self.dependencies[key] = tmp
                else:
                    self.dependencies[key] = []

    def add_prompt(self, element_name):
        if element_name == '':
            print("element_name is empty")
            return ""

        sub_prompt = ""

        ele = "element: "
        ele += element_name
        ele += '\n'
        sub_prompt += ele

        desc = "Description: "
        desc += self.name_to_desc[element_name]
        desc += '\n'
        sub_prompt += desc

        # func = "Function: "
        # func += name_to_func[element_name]
        # func += '\n'
        # sub_prompt += func

        dependency_list = self.dependencies[element_name]
        if not dependency_list:  # dependency 列表为空
            depe = "Dependency: No dependency, this UI element is in the main screen of the app. \n\n"
            sub_prompt += depe
        else:
            depe = "Dependency: this UI element could be interacted after "
            for i, element in enumerate(dependency_list):
                depe += element
                if i < len(dependency_list) - 1:
                    depe += " or "
            # print(depe)
            sub_prompt += depe
            sub_prompt += '\n\n'

        return sub_prompt

    def generate_prompts(self, prompts_path, use_comb=False, ele_group_strides=[2, 4, 6]):
        self.init_construct_map()  # 每个 element 的 name、description、function 存在字典中

        # print(len(name_to_desc)) # 一共 145 个不同的 element

        with open(self.api_tree_paths_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.extract_dependencies(data)  # 提取并存储每个 element 的上一级 dependency

        name_to_desc_list = list(self.name_to_desc.keys())

        if use_comb:
            # 生成所有 C(145, 2) 的组合
            ele_groups = list(combinations(name_to_desc_list, 2))
        else:
            ele_groups = []
            for stride in ele_group_strides:
                for i in range(len(name_to_desc_list) - (stride - 1)):
                    ele_groups.append(name_to_desc_list[i:i + stride])

        cnt = 0
        for ele_group in ele_groups:
            # print(f"-------------------------group_id = {cnt}-------------------------")
            # choice1, choice2 = pair
            # if choice1 == '' or choice2 == '':
            if '' in ele_group:
                print(f"one of the elements in {ele_group} is empty, continue")
                continue
            now_prompt_element = {}  # 存储当前的这个 prompt 写了哪些 element，避免重复写
            # print({f"cnt = {cnt}--------------------choice: {choice1}, {choice2}----------"})

            # prompt_prefix  共 C(145,2) 条 prompt
            prompt = f"Suppose you are a dataset annotator who is working on generating a series of tasks about the {self.app_name} APP on a smartphone. You are given a series of UI elements in this APP, which you could interact with by tapping, long tapping, edit, scroll, etc. You should generate as many specific tasks that could be executed by a virtual assistant on a smartphone as possible. Note that the tasks you generate must only involve these elements. \n\nUI elements in the {self.app_name} APP: \n\n"

            prompt_end = '''Please write down the tasks you would like to generate. You must use the following JSON format:

["<Task 1>", "<Task 2>", "<Task 3>"...]

Now please generate the specifc tasks. Notice that:
- You should be specific and avoid vague tasks. for example, you are forbidden to give tasks like "Send a message", instead, you should say "Send a message 'good morning' to Alice". Namely, you should be ensure your task descriptions are detailed, incorporating elements like names, accounts, phone numbers, and addresses.
- Focus on end-user actions rather than detailing every step or UI element involved. Your tasks should mimic natural language commands that a user might give to a virtual assistant like Siri. 
- **Please do not output anything else but the JSON content**'''
            for choice in ele_group:
                if choice not in now_prompt_element:
                    now_prompt_element[choice] = 1
                    prompt += self.add_prompt(choice)
                    # print(f"把 {choice1} 加入 prompt 的 element 中，开始添加从 root 到 {choice1} 路径上的所有 element")

                    value = data[choice]  # 当前 element 在 api_paths 中的值
                    for item in value:  # 遍历当前元素中的路径
                        for sub_item in item:  # 遍历路径中的每个元素
                            if isinstance(sub_item, dict) and 'name' in sub_item:  # 判断元素是否为字典且包含'name'字段
                                name = sub_item['name']  # 获取'name'字段的值
                                if name not in now_prompt_element:
                                    now_prompt_element[name] = 1
                                    prompt += self.add_prompt(name)
                                # print("now prompt: {prompt}")

            prompt += prompt_end
            # group = {"id": cnt, "prompt": prompt}
            self.all_prompts[cnt] = prompt
            cnt += 1

        with open(prompts_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_prompts, f, ensure_ascii=False, indent=4)

        print(f"数据已写入 json 文件中")

    def query_all_task_prompts(self, prompts_path, answers_path, model='gpt-3.5-turbo'):
        # all_prompts = tools.load_json_file(prompts_path)
        gpt = mtp.MultiProcessingQuery(answers_path, model=model, is_json=False)
        gpt.query_all_dicts(self.all_prompts, worker_num=8)

    @staticmethod
    def merge_original_and_preparation_tasks(app_name, model, pre_task_path, merge_prompt_path, merge_output_path):
        all_prompts = {}
        prompt = f"Suppose you are a dataset annotator who is working on generating a series of tasks about the {app_name} APP on a smartphone. Now I will give you a task and its corresponding preparation task and reason. Please merge the task and preparation task. The new merged task is required to be correct and complete. \n"
        with open(pre_task_path, 'r', encoding='utf-8') as file:
            cnt = 1
            for line in file:
                item = json.loads(line.strip())
                task = item.get("task", "")
                preparation_tasks = item.get("Preparation Tasks", "")
                reason = item.get("Reason", "")
                prompt = f"Suppose you are a dataset annotator who is working on generating a series of tasks about the {app_name} APP on a smartphone. Now I will give you a task and its corresponding preparation task and reason. Please merge the task and preparation task. The new merged task is required to be correct and complete. \n"
                prompt += f"Task: {task}\nPreparation Tasks:{preparation_tasks}\nReason:{reason}\n"
                prompt += """Please write down the tasks you would like to generate. You must use the following JSON format:

    ["<merged_task>"]

    Now please generate the specifc task, you just need generate one task. Please do not output anything else but the JSON content.
    """
                all_prompts[cnt] = prompt
                cnt += 1
                # if cnt >= 10: break

        with open(merge_prompt_path, 'w', encoding='utf-8') as f:
            json.dump(all_prompts, f, ensure_ascii=False, indent=4)

        print(f"merge prompt 已写入 {merge_prompt_path} ")
        gpt = mtp.MultiProcessingQuery(merge_output_path, model=model, is_json=False)
        gpt.query_all_dicts(all_prompts, worker_num=4)


class SolutionGenerator:
    def __init__(self, apis_folder_path, api_paths_file_path):
        self.apis_folder_path = apis_folder_path
        self.api_paths = tools.load_json_file(api_paths_file_path)

    def get_all_tasks(self, tasks_path):
        raw_task_answers = tools.load_jsonl_file(tasks_path)
        all_tasks = []
        for task_answer_data in raw_task_answers:

            if 'group_id' in task_answer_data.keys():
                # formats for file like data/tasks/q_a_task_05311700.jsonl
                task_id = task_answer_data['group_id']
                task_answer = task_answer_data['answer']
            else:
                # formats for file like data/tasks/tasks_0611.json
                task_id = list(task_answer_data.keys())[0]
                task_answer = list(task_answer_data.values())[0]
                task_answer = task_answer.replace('```json', '').replace('```', '')
                while task_answer.startswith('\n'):
                    task_answer = task_answer[1:]
                while task_answer.endswith('\n'):
                    task_answer = task_answer[:-1]
            try:
                task_list = ast.literal_eval(task_answer)
                if isinstance(task_list, dict):
                    task_list = list(task_list.values())[0]
                    task_list = ast.literal_eval(task_list)
            except:
                print(task_id, 'format error')
                continue
            all_tasks.extend(task_list)
        return all_tasks

    def format_all_apis(self, enable_dependency):
        apis_data = json.load(open(os.path.join(self.apis_folder_path, 'apis.json')))
        apis_description = ''
        api_names = []
        for _, v in apis_data.items():
            for e in v:
                if e["name"] == "" or e["name"] in api_names:
                    continue
                api_names.append(e["name"])
                name = e["name"]
                if enable_dependency:
                    dep = e['dependency']
                    semantic_dep = get_semantic_dependencies(dep)
                desc = e["desc"]
                func = e["func"]

                if enable_dependency:
                    apis_description += f'element: {name}\n\tDescription: {desc}\n\t Function: {func} \n\tDependency: {semantic_dep}. \n\n'
                else:
                    apis_description += f'element: {name}\n\tDescription: {desc}\n\t Function: {func} \n\n'
        print(f'Generated description for {len(api_names)} APIs')
        return apis_description

    def make_prompt(self, tasks, formatted_apis, app_name, first_state_apis=None):

        if first_state_apis is not None:
            first_ui_apis_statement = '\nThe first state of the app has the following important UI elements: \n'
            first_ui_apis = first_ui_apis_statement + '\t' + '\n\t'.join(first_state_apis) + '\n\n'
        else:
            first_ui_apis = ''
        formatted_tasks = '\n'.join([f'Task {i}: {tasks[i]}' for i in range(len(tasks))])

        prompt = f'''A {app_name} app in smartphone has the following important UI elements:

{formatted_apis}

You will be asked to complete tasks by writing scripts to manipulate the above elements.
In the script, except for the common python control flow (for, if-else, function def/calls, etc.), you can use the following APIs:
- tap(<element_selector>): tap on the element. Almost all elements can be taped. If an element's attribute checked=false or selected=false, tapping it can make it checked or selected, vice versa.
- long_tap(<element_selector>): long tap the element. 
- set_text(<element_selector>, <text>): set the text of the element to <text>. Only editable text fields can be set text.
- scroll(<element_selector>, <direction>): scroll the UI element in the specified direction, and direction is a str from "up", 'down", "left", "right". e.g. scroll($scroll_settings_page, "down"
- get_text(<element_selector>): return the text of the element as a string.
- get_attributes(<element_selector>): return the attributes of the element as a dict, dict keys include "selected", "checked", "scrollable", dict values are boolean. eg. get_attributes($files[3])["selected"].
- back(): close the current window


The <element_selector> primitive is used to select an element, possible ways of selection include:
- $<element id>, eg. $settings_button
- <element_list>[<idx>]: the idx-th in the element list. eg. $my_items[1]

The <element_list> primitive is used to select a list of elements, possible ways of selection include:
- <element_selector>: the items in the list element identified by <element_selector>. eg. $my_items
- <element_list>.match(<text or attribute dict>): the elements in the element list that match the given text or attribute dict. eg. $my_items.match("key words") or $my_items.match({{"selected": true}})
You can use len(<element_list>) to get the total number of items in an element list.
{first_ui_apis}

Now I will give you some tasks, you should return the python scripts to complete each task.
The tasks are:

{formatted_tasks}

Your answer should follow this JSON format:

{{
    "<task1>": "<script1>",
    "<task2>": "<script2>",
    ...
}}

**Note that the script is a string of python code and should only output the JSON content.**
'''
        return prompt

    def get_all_solution_query_prompts(self, app_name, tasks_path, solution_prompt_path, enable_dependency=True,
                                       max_task_per_prompt=8, first_state_apis=None):
        # raw output from GPT, in a list of dicts format
        if tasks_path.endswith('.jsonl'):
            all_tasks = self.get_all_tasks(tasks_path)
        # test tasks from a json file, in a list format
        else:
            all_tasks = tools.load_json_file(tasks_path)
        formatted_apis = self.format_all_apis(enable_dependency=enable_dependency)
        all_prompts = {}
        for i in range(0, len(all_tasks), max_task_per_prompt):
            if i + max_task_per_prompt > len(all_tasks):
                tasks = all_tasks[i:]
            else:
                tasks = all_tasks[i:i + max_task_per_prompt]

            prompt = self.make_prompt(tasks, formatted_apis, app_name, first_state_apis=first_state_apis)
            all_prompts[i] = prompt
        tools.dump_json_file(solution_prompt_path, all_prompts)
        print(
            f'**Loaded {len(all_tasks)} tasks. Generated {len(all_prompts)} prompts, {max_task_per_prompt} tasks per prompt.**')
        return all_prompts

    def query_all_solution_prompts(self, app_name, tasks_path, solution_prompt_path, solution_answers_path,
                                   max_task_per_prompt=5, model='gpt-3.5-turbo'):
        all_prompts = self.get_all_solution_query_prompts(app_name, tasks_path, solution_prompt_path,
                                                          max_task_per_prompt)
        gpt = mtp.MultiProcessingQuery(solution_answers_path, model=model, is_json=False)
        gpt.query_all_dicts(all_prompts, worker_num=8)


class PreTaskGenerator:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def define_prompt(self, tasks):
        prompt = f"""As a task orchestrator for an Android notes application, your role is to ensure that the UI state is suitable for executing various main tasks by providing preparation tasks. The preparation tasks aim to identifyand resolve obstacles or conflicts that might appear when attempting to complete the main task directly on the application. Common issues involve: resolving naming conflicts, not found items, creating or deleting items, navigating to specific screens, etc.
Execution flow:
1.User always starts from the home page of the application.
2.Then the user attempts to complete the preparation tasks you provide.
3.Finally user proceeds to the main task completion.

Instructions:
- Given a main task, generate preparation tasks necessary to set up the UI state for its execution.
- Preparation tasks and main tasks should not have duplicate actions. 
- Do not include checks for network connectivity, system resources, or other system-related verifications, assuming the system is in a perfect state for UI operations.

Format: Markdown
# Main Task: [Description of the main task],
# Preparation Tasks:
## Name: [Provide a clear and concise definition of a preparation task according to your reasoning. Make sure it is concrete.]
- Reason: [Give reasons why the preparation task is needed required. Explain your reasoning step by step in clear to follow sentences.]
..

Example:

Generate Preparation tasks for the given list of main tasks:

Task 1: 'Rename the note titled "Old Title" to "New Title".'
Task 2: 'Print the note titled "To Print".'
Task 3: 'Exit the settings page.'
Task 4: 'Make links and emails inside my note clickable.'
Task 5: 'Remove the done items from the current checklist.'
Task 6: 'Open the application information in the settings'

Output: 
# Task 1: 'Rename the note titled "Old Title" to "New Title" '
# Preparation Tasks:
## Name: Create the note "Old Title" if it doesn't exist.
- Reason: This task guarantees the note exists, so it can be renamed during execution of the main task.
## Name: If a note with the title "New Title" already exists, rename the note title to something else.
- Reason: This task guarantees there is no note with with the same title, to prevent naming conflicts during execution of the main task.

# Task 2: Create a new note with the title "Grocery List"
# Preparation Tasks:
## Name: If a note with the title "Grocery List" already exists, rename the note title to something else.
- Reason: This task guarantees the note does not exists, so it can be created during execution of the main task. 

# Task 3: 'Exit the settings page.'
# Preparation Tasks:
## Name: Open the settings page if the current page is not the settings page.
- Reason: This task guarantees that the current page of the application is the settings page, so it can be exitted during execution of the main task.

# Task 4: 'Make links and emails inside my note clickable.'
# Preparation Tasks:
## Name: Open a note or create one if there is none
- Reason: This task guarantees a note exists and it is opened. 
## Name: Write links and emails inside the opened note
- Reason: This task guarantees that links and emails exist inside the opened note, so they can be made clickable during execution of the main task. 

# Task 5: 'Remove the done items from the current checklist.'
# Preparation Tasks:
## Name: Open a checklist note or create one if there is none
- Reason: This task guarantees a checklist note exists and it is opened. 
## Name: Create checklist items and mark them as done 
- Reason: This task guarantees that the checklist note contains items that are marked as done, so the main task could remove them.

# Main Task 6: 'Open the application information in the settings'
# Preparation Tasks: None
- Reason: As we always start from the home page, the settings page can be navigated easily from the main task without the need of prior steps.  

Now generate preparation tasks for the given list of main tasks and explain the reasons for their existence step by step.

{tasks}
Please output nothing else but the specified Markdown format.
Output:
"""
        example = """
- Main Task 1: Create a new note with the title "Grocery List" 
  Preparation Task: Delete the note titled "Grocery List"
  Reason: This ensures there are no existing notes with the title "Grocery List," preventing naming conflicts.

- Main Task 2: Export the note titled "Meeting Notes" as a file 
  Preparation Task: Ensure the note titled "Meeting Notes" is open
  Reason: Opening the note ensures it is ready for export, avoiding any UI issues related to note selection.

- Main Task 3: Rename the note titled "Old Title" to "New Title" 
  Preparation Task: Ensure the note titled "Old Title" exists
  Reason: Verifying the existence of "Old Title" ensures the rename operation can proceed without errors.

- Main Task 4: Delete the note titled "Unwanted Note" 
  Preparation Task: Ensure the note titled "Unwanted Note" exists
  Reason: Ensuring the note exists allows for its deletion, avoiding errors from attempting to delete a non-existent note.

- Main Task 5: Lock the note titled "Private Note" 
  Preparation Task: Ensure the note titled "Private Note" is open
  Reason: Opening the note prepares it for locking, preventing issues related to accessing the note.

- Main Task 6: Close the search box 
  Preparation Task: Open the search box
  Reason: Opening the search box is necessary for it to be closed, setting up the UI for the main task.

- Main Task 7: Create a shortcut for the current note 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note ensures it is recognized as the current note, allowing for the creation of a shortcut.

- Main Task 8: Edit the content of the current note to "Discuss budget allocations" 
  Preparation Task: Open the note to be edited
  Reason: Ensuring the note is open allows for content editing, preventing issues with accessing the note.

- Main Task 9: Open the about page of the app 
  Preparation Task: Navigate to the main menu
  Reason: Accessing the main menu provides a direct path to the about page, ensuring smooth navigation.

- Main Task 10: Delete the current note 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note sets it as the current note, allowing for its deletion.

- Main Task 11: Scroll to the next note or checklist item 
  Preparation Task: Ensure a note or checklist is open
  Reason: Having a note or checklist open ensures there is content to scroll through, preventing UI errors.

- Main Task 12: Print the current note 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note sets it as the current note, enabling the print function.

- Main Task 13: Lock the current note with a password 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note prepares it for locking, ensuring the UI is ready for password entry.

- Main Task 14: Save the edits made to the current note 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note allows for edits to be made and saved, avoiding issues with accessing the note.

- Main Task 15: Scroll down the settings page 
  Preparation Task: Open the settings page
  Reason: Opening the settings page is necessary for scrolling, ensuring the UI is in the correct state.

- Main Task 16: Set the font size of the note to 175% 
  Preparation Task: Open a note
  Reason: Ensuring a note is open allows for font size adjustments, preventing UI inconsistencies.

- Main Task 17: Switch to the next checklist item 
  Preparation Task: Open a checklist
  Reason: Having a checklist open allows for switching between items, ensuring the UI is ready.

- Main Task 18: Increase the font size to 250% 
  Preparation Task: Open a note
  Reason: Opening a note is necessary to adjust the font size, ensuring the UI is in the correct state.

- Main Task 19: Add a new checklist item titled "Grocery Shopping" 
  Preparation Task: Open a checklist
  Reason: Ensuring a checklist is open allows for adding new items, preventing UI errors.

- Main Task 20: Search for the word "meeting" in the current note 
  Preparation Task: Open the note to be used as the current note
  Reason: Opening the note sets it as the current note, enabling the search function.

- Main Task 21: Print the note titled "To Print" 
  Preparation Task: Ensure the note titled "To Print" is open
  Reason: Opening the note prepares it for printing, ensuring the UI is ready.

- Main Task 22: Remove done items from the checklist titled "To Do" 
  Preparation Task: Open the checklist titled "To Do"
  Reason: Opening the checklist ensures it is ready for item removal, preventing UI issues.

- Main Task 23: Scroll down the settings page 
  Preparation Task: Open the settings page
  Reason: Opening the settings page is necessary for scrolling, ensuring the UI is in the correct state.

- Main Task 24: Exit the settings page 
  Preparation Task: Open the settings page
  Reason: Opening the settings page ensures it is ready to be exited, preventing navigation issues.

- Main Task 25: Create a shortcut for the note titled "Frequently Used" 
  Preparation Task: Ensure the note titled "Frequently Used" is open
  Reason: Opening the note prepares it for shortcut creation, ensuring the UI is ready.

- Main Task 26: Set the place cursor to the end of the note 
  Preparation Task: Open a note
  Reason: Opening a note is necessary to set the cursor position, ensuring the UI is in the correct state.

- Main Task 27: Open the note titled "Grocery List" 
  Preparation Task: Ensure the note titled "Grocery List" exists
  Reason: Verifying the existence of "Grocery List" ensures the note can be opened without errors.

- Main Task 28: Make links and emails clickable in the settings 
  Preparation Task: Open the settings page
  Reason: Opening the settings page is necessary to adjust the clickability settings, ensuring the UI is ready.

- Main Task 29: Create a new note titled "Meeting Agenda" 
  Preparation Task: Delete any existing note titled "Meeting Agenda"
  Reason: This ensures there are no existing notes with the title "Meeting Agenda," preventing naming conflicts.

- Main Task 30: Exit the settings page 
  Preparation Task: Open the settings page
  Reason: Opening the settings page ensures it is ready to be exited, preventing navigation issues
"""
        return prompt

    def get_notes_titles(self):
        data = []
        with open(self.input_file, 'r') as file:
            for line in file:
                item = json.loads(line.strip())
                title = item['items'][0]["content"]
                title = title.strip("\n").strip("'\\'")
                data.append(title)
        return data

    def process_batch(self, batch):
        formatted = ""
        for i, item in enumerate(batch):
            formatted += f"Task {i + 1}: {item} \n"
        return formatted

    def extract_task_info(self, task_description):
        lines = [line for line in task_description.strip().splitlines() if line]
        if len(lines) < 3:
            return None
        main_task = lines[0].replace("**", " ").split(": ", 1)[1].strip()
        preparation_task = lines[1].replace("**", " ").split(": ", 1)[1].strip()
        reason = lines[2].replace("**", " ").split(": ", 1)[1].strip()
        return {
            "task": main_task,
            "preparation": preparation_task,
            "reason": reason
        }

    def parse_task_descriptions(self, task_string):
        task_descriptions = task_string.split("\n\n")
        result = []
        for task_desc in task_descriptions:
            info = self.extract_task_info(task_desc.strip())
            if info is not None:
                result.append(info)
        return result

    def read_result(self, key="batch300"):
        data = self.load_jsonl_file(self.output_file)
        for entry in data:
            if key in entry:
                return entry[key]

    def load_jsonl_file(self, file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line.strip()) for line in file]

    def generate_tasks(self):
        data = self.get_notes_titles()  # data 来源 /data/notes_train_data.jsonl
        batch_size = 50
        dict = {}
        llm = MultiProcessingQuery(self.output_file)

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            formatted = self.process_batch(batch)
            prompt = self.define_prompt(formatted)
            dict[f"batch{i}_{len(batch)}"] = prompt

        llm.query_all_dicts(dict, 8)


class FormatOrganizer:

    def __init__(self) -> None:
        pass

    @staticmethod
    def convert_str_to_dict(str_data):
        if isinstance(str_data, dict):
            return str_data
        str_data = str_data.replace('```json', '').replace('```', '')
        str_data = str_data.replace('```dict', '').replace('```', '')
        str_data = str_data.replace('```python', '').replace('```', '')
        while str_data.startswith('\n'):
            str_data = str_data[1:]
        while str_data.endswith('\n'):
            str_data = str_data[:-1]
        try:
            dict_data = ast.literal_eval(str_data)
        except:
            print(f'format error, {str_data} can not be converted to dict.')
            dict_data = {}
        return dict_data

    @staticmethod
    def form_task_solution_jsonl(solution_prompts_path, solution_answers_path, output_jsonl_path):
        solution_prompts = tools.load_json_file(solution_prompts_path)
        solutions = tools.load_jsonl_file(solution_answers_path)
        output_jsonl = []
        for solution_data in solutions:
            # solution_data: {'id': 'xxx'}
            if not list(solution_data.values())[0]:
                print(f'{solution_data} Empty solution')
                continue
            solution_dict = FormatOrganizer.convert_str_to_dict(list(solution_data.values())[0])
            if solution_dict == {}:
                continue
            solutions = list(solution_dict.values())

            prompt_id = list(solution_data.keys())[0]
            prompt = solution_prompts[prompt_id]
            pattern = r'Task \d+: ([^\n]+)'
            matches = re.findall(pattern, prompt)
            tasks = [match.strip() for match in matches]
            if len(tasks) != len(solutions):
                print(f'Error: prompt {prompt_id} has {len(tasks)} tasks but {len(solutions)} solutions')
                # import pdb;pdb.set_trace()
                continue
            output_jsonl.extend([{'task': task, 'solution': solution} for task, solution in zip(tasks, solutions)])

        tools.dump_jsonl_file(output_jsonl, output_jsonl_path)
        print(f'**Formated {len(output_jsonl)} tasks and solutions into {output_jsonl_path}**')

    @staticmethod
    def format_debug_message_data(prompts, answers, output_jsonl_path):
        all_prompts = tools.load_json_file(prompts)
        all_answers = tools.load_jsonl_file(answers)
        task_solutions = []
        for i, answer in enumerate(all_answers):
            # answer: {'id': 'xxx'}
            error_id = list(answer.keys())[0]
            prompt = all_prompts[error_id]
            task_content = re.search(r"The task is:\s+(.*?)\n\nCurrent UI", prompt, re.DOTALL)
            if task_content:
                task = task_content.group(1).strip()
                if task.startswith('\n'):
                    task = task[1:]
            else:
                print("Task content not found.")
                continue
            solution_dict = FormatOrganizer.convert_str_to_dict(list(answer.values())[0])
            if 'Script' in solution_dict.keys():
                solution = solution_dict['Script']
            else:
                print(f'Error: Script not found in solution {error_id}')
                continue
            task_solutions.append({"task": task, "solution": solution})
        tools.dump_jsonl_file(task_solutions, output_jsonl_path)
        print(f'**Formated {len(task_solutions)} tasks and solutions into {output_jsonl_path}**')

    @staticmethod
    def format_train_message_data(output_jsonl_path, train_data_write_to_path):
        # 打开输入文件，读取数据并转换格式
        with open(output_jsonl_path, 'r', encoding='utf-8') as infile, open(train_data_write_to_path, 'w',
                                                                            encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line.strip())
                # 构造新的格式
                new_data = {
                    "items": [
                        {"role": "user", "content": f"'{data['task']}'"},
                        {"role": "assistant", "content": data['solution']}
                    ],
                    "system": ""
                }
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        print("Finish formatting tasks and solutions for training.")

    @staticmethod
    def pre_task_format(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            task_counter = 1
            for line in infile:
                data = json.loads(line)
                for key, value in data.items():
                    if value is None: continue
                    tasks = value.split('\n\n')
                    for task in tasks:
                        task_number = f"{task_counter}"
                        prep_task_match = re.search(r'Preparation Tasks:\n(## Name: (.*?)\n- Reason:.*?)+', task,
                                                    re.DOTALL)
                        if prep_task_match:
                            prep_tasks = re.findall(r'## Name: (.*?)\n- Reason:', prep_task_match.group(0))
                            if prep_tasks:
                                for prep_task in prep_tasks:
                                    formatted_task = prep_task.replace('\"', '\'')
                                    result = f'{{"{task_number}": "[\\"{formatted_task}\\"]"}}'
                                    outfile.write(result + '\n')
                        task_counter += 1

    @staticmethod
    def merge_pre_task_with_origin_task(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            task_counter = 1
            for line in infile:
                data = json.loads(line)
                for key, value in data.items():
                    tasks = value.split('\n\n')
                    for task in tasks:
                        task_number = f"{task_counter}"

                        # 提取 Preparation Tasks
                        prep_task_match = re.search(r'Preparation Tasks:\n(## Name: (.*?)\n- Reason:.*?)+', task,
                                                    re.DOTALL)
                        prep_tasks = []
                        if prep_task_match:
                            prep_tasks = re.findall(r'## Name: (.*?)\n- Reason:', prep_task_match.group(0))

                        # 提取 Task
                        task_match = re.search(r'# Task \d+: (.*?)\n', task)
                        if task_match:
                            task_title = task_match.group(1)
                            # 拼接 Preparation Tasks 和 Task
                            if prep_tasks:
                                prep_tasks_str = ' and '.join(prep_tasks)
                                formatted_task = f"{prep_tasks_str} And {task_title}".replace('\"', '\'')
                            else:
                                formatted_task = task_title.replace('\"', '\'')

                            result = f'{{"{task_number}": "[\\"{formatted_task}\\"]"}}'
                            outfile.write(result + '\n')
                        task_counter += 1
        print(f"pre_task与origin_task已合并，已写入到文件 {output_file}")

    @staticmethod
    def process_and_make_readable(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                data = json.loads(line.strip())
                for key, task in data.items():
                    if not isinstance(task, str):
                        continue
                    processed_tasks = []
                    lines = task.split('\n')
                    current_task = {}
                    for line in lines:
                        if line.startswith('# Task'):
                            if current_task:
                                processed_tasks.append(current_task)
                                current_task = {}
                            current_task['task'] = line.split(': ')[1]
                        elif line.startswith('## Name:'):
                            current_task['Preparation Tasks'] = line.split(': ')[1]
                        elif line.startswith('- Reason:'):
                            current_task['Reason'] = line.split(': ')[1]
                    if current_task:
                        processed_tasks.append(current_task)

                    for processed_task in processed_tasks:
                        # Ensure "Preparation Tasks" field exists
                        if "Preparation Tasks" not in processed_task:
                            processed_task["Preparation Tasks"] = "None"

                        # Remove extra single and double quotes from the "task" field if it exists
                        if "task" in processed_task:
                            task = processed_task["task"]
                            if task.startswith("'") and task.endswith("'"):
                                task = task[1:-1]
                            elif task.startswith('"') and task.endswith('"'):
                                task = task[1:-1]
                            processed_task["task"] = task

                        # Reorder the dictionary to ensure "Preparation Tasks" is before "Reason"
                        reordered_task = {"task": processed_task["task"]} if "task" in processed_task else {}
                        reordered_task["Preparation Tasks"] = processed_task["Preparation Tasks"]
                        if "Reason" in processed_task:
                            reordered_task["Reason"] = processed_task["Reason"]

                        # Write the modified task to the output file
                        json.dump(reordered_task, outfile, ensure_ascii=False)
                        outfile.write('\n')
        print(f"Preparation Tasks have been Formatted to {output_file}")


class HardTasksGenerator:
    def __init__(self, task_path, output_path, model_name):
        self.task_path = task_path
        self.output_path = output_path
        self.model_name = model_name

    def get_task_prompts(self):
        all_task_query_prompts = {}
        id = 0
        with open(self.task_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                # if id == 5: print(data)
                for key, value in data.items():
                    prompt = f'''Suppose you are using a Notes app where you can complete tasks by interacting with the app. Here are some simple tasks you can perform:

{value}

Please think of some complex tasks by combining the above tasks or creating more complex control flows. Use the following JSON format:
```json
"[\"complex task 1\", \"complex task 2\", \"complex task 3\", \"complex task 4\", \"complex task 5\", \"complex task 6\", \"complex task 7\", \"complex task 8.\"]"
```

When generating the specific tasks, ensure the following:
- You should be specific and avoid vague tasks. for example, you are forbidden to give tasks like "Send a message", instead, you should say "Send a message 'good morning' to Alice". Namely, you should be ensure your task descriptions are detailed, incorporating elements like names, accounts, phone numbers, and addresses.
- Focus on end-user actions rather than detailing every step or UI element involved. Your tasks should mimic natural language commands that a user might give to a virtual assistant like Siri, but should be more complex. 
- **Please do not output anything else but the JSON content** Ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).'''
                    # if id == 5: print(prompt)
                    all_task_query_prompts[id] = prompt
                    id += 1
        querier = MultiProcessingQuery(self.output_path, self.model_name, is_json=False)
        querier.query_all_dicts(all_task_query_prompts, worker_num=8)

    def format_tasks(self):
        with open(self.output_path, 'r') as file:
            lines = file.readlines()
        converted_lines = []
        for line in lines:
            data = json.loads(line)
            key = list(data.keys())[0]
            tasks = data[key]

            # tasks = tasks.replace("```json\n", "").replace("\n```", "")
            # tasks_list = json.loads(tasks)
            # tasks = re.sub(r'```json\n|\n```', '', tasks)
            # tasks_list = json.loads(tasks)
            tasks = tasks[8:-4]

            new_data = {key: tasks}
            converted_lines.append(json.dumps(new_data))

        with open(self.output_path, 'w') as file:
            for line in converted_lines:
                file.write(line + '\n')


if __name__ == '__main__':
    args = parseargs()

    if args.mode == 'doc_format':
        solver = APIPathSolver('output/notes0503')
        api_paths = solver.add_action_type_for_dependencies()
        # print(json.dumps(api_paths, indent=4))
        tools.dump_json_file('tmp/api_paths.json', api_paths)

    if args.mode == 'task':
        # TaskGenerator 传入 api 文档和 api_path 进行初始化
        task_generator = TaskGenerator(app_name='Notes', apis_path='output/notes0503/apis.json',
                                       api_tree_paths_path='tmp/api_paths.json')
        # 生成 original tasks 的 prompt，下列函数中传入的路径为：输出 prompt 的路径
        task_generator.generate_prompts('pipeline_data/origin_task_prompt.json', use_comb=False,
                                        ele_group_strides=[2, 4, 6, 8])
        # 对于上一步生成的 prompt，让 GPT 生成 original task，下列函数中传入的路径为：prompt 的路径，生成的 original task 的路径
        # 生成的 original task 的格式，例如：{"329": "[\"Tap more options on the current note\", \"Tap more options on the current checklist\", \"Open settings\", \"Go to customize colors\", \"Set the primary color of the app to blue\", \"Set the app icon color to green\", \"Set the theme color to auto light/dark mode\", \"Set the theme color to light\", \"Set the theme color to dark\", \"Set the theme color to dark red\", \"Set the theme color to white\"]"}
        task_generator.query_all_task_prompts('pipeline_data/origin_task_prompt.json',
                                              'pipeline_data/gpt_origin_task.jsonl', model='gpt-4o')

    if args.mode == 'solution':
        # SolutionGenerator 传入 output 与 api_path 进行初始化
        solution_generator = SolutionGenerator(apis_folder_path='output/notes0503',
                                               api_paths_file_path='tmp/api_paths.json')
        solution_generator.get_all_solution_query_prompts(app_name='Notes',
                                                          tasks_path='pipeline_data/origin_gpt_gen_task.jsonl',
                                                          solution_prompt_path='tmp/origin_task_solution_prompts.json',
                                                          max_task_per_prompt=5)
        # 生成 original task 的 solution；下列函数中传入的路径为：original task 的路径，让 GPT 生成 solution 的 prompt 的路径，GPT 生成的 solution 的路径
        # solution 文件的格式，例如：{"485": "```json\n{\n    \"Task 0\": \"tap($more_options_checklist)\\ntap($open_file)\",\n    \"Task 1\": \"tap($more_options_note)\",\n    \"Task 2\": \"tap($more_options_checklist)\",\n    \"Task 3\": \"tap($settings)\\ntap($set_show_word_count)\",\n    \"Task 4\": \"tap($settings)\\ntap($set_make_links_and_emails_clickable)\"\n}\n```"}
        if args.query:
            solution_generator.query_all_solution_prompts(app_name='Notes',
                                                          tasks_path='pipeline_data/origin_gpt_gen_task.jsonl',
                                                          solution_prompt_path='pipeline_data/origin_task_solution_prompts.json',
                                                          solution_answers_path='pipeline_data/origin_task_solution_answers.jsonl',
                                                          model='gpt-4o')
            # 需要将 task 和 solution 对应起来，这一步 format 处理之后输出格式例如：{"task": "Sort the current checklist items.", "solution": "tap($more_options_checklist); tap($sort_checklist_items)"}
            FormatOrganizer.form_task_solution_jsonl('pipeline_data/origin_task_solution_prompts.json',
                                                     'pipeline_data/origin_task_solution_answers.jsonl',
                                                     'pipeline_data/origin_task_solution_format.jsonl')
            # format 到微调的格式，为方便直观检查 task / solution 的质量，这一步先不用，用上一步 format 的格式即可
            FormatOrganizer.format_train_message_data('pipeline_data/origin_task_solution_format.jsonl',
                                                      'pipeline_data/origin_task_solution_format_tmp.jsonl')

    if args.mode == 'pre_task':
        # PreTaskGenerator 初始化传入两个路径：调整好格式的 original task，输出生成的 pre task
        pre_task_generator = PreTaskGenerator(input_file="pipeline_data/origin_task_solution_format_tmp.jsonl",
                                              output_file="pipeline_data/pre_gpt_gen_task.jsonl")
        # 调用类的方法来生成 pre task
        pre_task_generator.generate_tasks()
        # 这一步 format 之后的格式为：
        # {"task": "Confirm the changes made to \"Travel Details\" note", "Preparation Tasks": "Open the note titled \"Travel Details\"", "Reason": "This task guarantees that the note is opened so changes can be confirmed during the execution of the main task."}
        FormatOrganizer.process_and_make_readable('pipeline_data/pre_gpt_gen_task.jsonl',
                                                  'pipeline_data/pre_task_format.jsonl')
        # 这一步的 merge 是简单的 "and" 连接
        # FormatOrganizer.merge_pre_task_with_origin_task("data/notes_pre_task.jsonl", "data/notes_merge_pre_origin_task.jsonl")
        # 调用以下方法来合并 original task 和 preparation task
        TaskGenerator.merge_original_and_preparation_tasks('Notes', 'gpt-4o', 'pipeline_data/pre_task_format.jsonl',
                                                           'pipeline_data/merge_origin_pre_task_prompt.jsonl',
                                                           'pipeline_data/merge_origin_pre_gpt_gen_task.jsonl')

    if args.mode == 'pre_task_solution':
        # FormatOrganizer.pre_task_format(input_file='pipeline_data/pre_gpt_gen_task.jsonl', output_file='pipeline_data/pre_only_task_format.jsonl')
        # solution_generator = SolutionGenerator(apis_folder_path='output/notes0503', api_paths_file_path='tmp/api_paths.json')
        # solution_generator.query_all_solution_prompts(app_name='Notes', tasks_path='pipeline_data/pre_only_task_format.jsonl', solution_prompt_path='pipeline_data/pre_task_solution_prompts.json', solution_answers_path='pipeline_data/pre_task_solution_answers.jsonl', model='gpt-4o')
        # FormatOrganizer.form_task_solution_jsonl('pipeline_data/pre_task_solution_prompts.json', 'pipeline_data/pre_task_solution_answers.jsonl', 'pipeline_data/pre_task_solution_format.jsonl')
        # 生成合并 origin pre 后的 solution
        solution_generator = SolutionGenerator(apis_folder_path='output/notes0503',
                                               api_paths_file_path='tmp/api_paths.json')
        solution_generator.query_all_solution_prompts(app_name='Notes',
                                                      tasks_path='pipeline_data/merge_origin_pre_gpt_gen_task.jsonl',
                                                      solution_prompt_path='pipeline_data/merge_origin_pre_task_solution_prompts.json',
                                                      solution_answers_path='pipeline_data/merge_origin_pre_task_solution_answers.jsonl',
                                                      model='gpt-4o')
        FormatOrganizer.form_task_solution_jsonl('pipeline_data/merge_origin_pre_task_solution_prompts.json',
                                                 'pipeline_data/merge_origin_pre_task_solution_answers.jsonl',
                                                 'pipeline_data/merge_origin_pre_task_solution_format.jsonl')

    if args.mode == 'hard_task':
        task_generator = HardTasksGenerator('pipeline_data/origin_gpt_gen_task.jsonl',
                                            'pipeline_data/hard_gpt_gen_task.jsonl', 'gpt-4o')
        task_generator.get_task_prompts()
        task_generator.format_tasks()

    if args.mode == 'hard_task_solution':
        solution_generator = SolutionGenerator(apis_folder_path='output/notes0503',
                                               api_paths_file_path='tmp/api_paths.json')
        solution_generator.query_all_solution_prompts(app_name='Notes',
                                                      tasks_path='pipeline_data/hard_gpt_gen_task.jsonl',
                                                      solution_prompt_path='pipeline_data/hard_task_solution_prompts.json',
                                                      solution_answers_path='pipeline_data/hard_task_solution_answers.jsonl',
                                                      model='gpt-4o')
        FormatOrganizer.form_task_solution_jsonl('pipeline_data/hard_task_solution_prompts.json',
                                                 'pipeline_data/hard_task_solution_answers.jsonl',
                                                 'pipeline_data/hard_task_solution_format.jsonl')
        # FormatOrganizer.format_train_message_data('data/notes_train_data_hard_0618.jsonl', 'data/notes_format_train_data_hard.jsonl')

