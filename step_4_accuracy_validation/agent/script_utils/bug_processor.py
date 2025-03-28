import re
import agent.environment as environment
import tools as tools
from lxml import etree
from agent.script_utils.gen_dependency_tree import get_semantic_dependencies
from agent.script_utils.api_doc import ApiDoc
from agent.script_utils.err import ActionError, APIError, XPathError, NotFoundError

from agent.script_utils.solution_generator import ACTIONS_DSL_PROMPT_DESCRIPTION

class BugProcessor:

  def __init__(self, app_name, log_path, error_log_path, task, raw_solution,
               apis_path, api_xpath_file):
    self.app_name = app_name
    self.raw_log = tools.load_yaml_file(log_path)
    self.error_log = tools.load_json_file(error_log_path)
    self.apis = tools.load_json_file(apis_path)
    self.task = task
    self.raw_solution = raw_solution
    self.api_xpaths = tools.load_json_file(api_xpath_file)

  def _get_view_without_id(self, view):
    modified_view = re.sub(r" id='\d+'", '', view)
    return modified_view

  def get_action_desc(self, action_type, selected_element, input_text=None):
    if selected_element:
      selected_element = self._get_view_without_id(selected_element)
      if selected_element.lower() == 'crashed':
        return 'Can not find the UI element. '
    if 'go back' in action_type.lower():
      return 'back()'
    if action_type.lower() in ['match', 'index', 'len']:
      return 'select a UI element'
    if action_type in ['touch', 'long_touch', 'select']:
      action_desc = f"{action_type}: {selected_element}"
    elif action_type.lower() in ['scroll up', 'scroll down']:
      action_desc = f"{action_type}"
    elif action_type == 'set_text':
      action_desc = f"Touch: {selected_element} Input_text: {input_text}"
    else:
      action_desc = f"{action_type}: {selected_element}"
    return action_desc

  def format_all_apis(self, enclude_function=False):
    apis_data = self.apis
    apis_description = ''
    api_names = []
    for _, v in apis_data.items():
      for e in v:
        if e["name"] == "" or e["name"] in api_names:
          continue
        api_names.append(e["name"])
        name = e["name"]
        dep = e['dependency']
        desc = e["desc"]
        func = e["func"]
        semantic_dep = get_semantic_dependencies(dep)
        if not enclude_function:
          apis_description += f'element: {name}\n\tDescription: {desc}\n\tDependency: {semantic_dep}. \n\n'
        else:
          apis_description += f'element: {name}\n\tDescription: {desc}\n\t Function: {func} \n\tDependency: {semantic_dep}. \n\n'
    print(f'Generated description for {len(api_names)} APIs')
    return apis_description

  def reorganize_log(self):
    records = self.raw_log['records']
    reorganized_records = []
    log_str = ''
    for record_id, ui_record in enumerate(records):
      # TODO: this is specific to the notes app, please modify it accordingly
      if (record_id == 0 and ui_record['Action'] == 'go back') or (
          record_id == 1 and ui_record['Action'] == 'scroll DOWN'):
        continue

      ui_apis = {}
      ui_state = ui_record['State']
      for api_name, api_xpath in self.api_xpaths.items():

        root = etree.fromstring(ui_state)
        eles = root.xpath(api_xpath)
        if not eles:
          continue
        ele_desc = etree.tostring(eles[0], pretty_print=True).decode(
            'utf-8')  # only for father node
        id_str = re.search(r' id="(\d+)"', ele_desc).group(1)

        if '_list' in api_name:
          str_elements = '\n\t'.join([
              etree.tostring(elem, pretty_print=True).decode('utf-8')
              for elem in eles
          ])
          api_desc = f'{api_name}: \n\t\t{str_elements}'
        else:
          api_desc = api_name
        id = int(id_str)
        ui_apis[id] = api_desc

      # iterate over ui_apis to get the order of apis
      ui_apis_ordered = []
      for id in sorted(ui_apis.keys()):
        ui_apis_ordered.append(ui_apis[id])
      ui_apis_str = '\n\t'.join(ui_apis_ordered)
      log_str += f"UI {record_id}: \n\t{ui_apis_str}\n"

      if 'statement' in ui_record.keys():
        if ui_record['Action'] == 'match' or ui_record['Action'] == 'index':
          match_result_list = ui_record['statement'].split('->')
          match_result = match_result_list[1].strip()
          code_statement = f"Currently executing code: {match_result_list[0]}, matched UI element XPath: {match_result}\n\n"
        elif ui_record['statement'] == 'len':
          code_statement = f"Currently executing code: {ui_record['statement']}, the result is the length of the UI element list\n\n"
        else:
          code_statement = f"Currently executing code: {ui_record['statement']}\n\n"
        log_str += code_statement
      if ui_record['Choice'] is not None:
        choice_id = ui_record['Choice']
        if isinstance(choice_id, int):
          tree = etree.HTML(ui_state)
          element = tree.xpath(f"//*[@id='{str(choice_id)}']")[0]

          target_element_desc = etree.tostring(
              element, method="html", pretty_print=True).decode('utf-8')
          action_desc = self.get_action_desc(ui_record['Action'],
                                             target_element_desc,
                                             ui_record['Input'])
        else:
          action_desc = self.get_action_desc(ui_record['Action'],
                                             selected_element=choice_id,
                                             input_text=ui_record['Input'])
      else:
        action_desc = self.get_action_desc(ui_record['Action'],
                                           selected_element=None,
                                           input_text=ui_record['Input'])

      log_str += f"Detailed UI action: {action_desc}\n\n"

      reorganized_records.append({
          'ui_apis': ui_apis_ordered,
          'ui_action': ui_record['Action']
      })

    return log_str

  def make_prompt(self, enclude_element_function=False, include_log_info=True):
    if include_log_info:
      ui_actions_log = self.reorganize_log()
      log_info = f'''
Script execution detailed log (all the available UI elements on each UI, the code line in the script that is currently executing, and the detailed action on each UI):
{ui_actions_log}
'''
    else:
      log_info = ''

    # if 'verifier' in error info, indicate llm that this statement is not supported, suggests to find another way to implement the same functionality
    if 'verifier' in self.error_log['error']:
      error_info = 'The statement is not supported by the verifier, please find another way to implement the same functionality.'
    else:
      error_info = self.error_log['error']

    prompt = f'''Suppose you are a mobile app testing expert who is working on testing the function of the {self.app_name} app on a smartphone, you are given a python-style script to complete a specific task, but you met bug when executing the script, you should try to fix it. You are provided with:

Task: 
{self.task} 

Original script of the task: 
```
{self.raw_solution}
```
{log_info}
Error: 
{error_info}

The script line that caused the error: 
{self.error_log['error_line_in_original_script']}

The above is the detailed information about the bug you encountered.
You are required to re-generate the script to complete the task. The script should be python-style code, and you can use the following UI elements to interact with the app: 

{self.format_all_apis(enclude_function=enclude_element_function)} 

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

Now please return the corrected script to complete the task. Your answer should in the following format:
{{
    'Reasoning': '<the reason why the bug occurs>',
    'Script': '<the corrected script>',
    'Explanation': '<the explanation of the corrected script>'
}}'''

    return prompt


class BugProcessorV2:

  def __init__(self, app_name, log_path, error_log_path, task: str, raw_solution: str, ele_data_path, doc: ApiDoc):
    self.app_name = app_name
    self.raw_log = tools.load_yaml_file(log_path)
    self.error_log = tools.load_json_file(error_log_path)
    self.ele_data = tools.load_json_file(ele_data_path)
    self.task = task
    self.raw_solution = raw_solution
    self.doc = doc

  def _get_view_without_id(self, view):
    modified_view = re.sub(r" id='\d+'", '', view)
    return modified_view

  def _get_ui_elements_of_stuck_ui(self):
    '''
    get all the UI elements from the UI page where the bug happens, and return them in the order of appearance in the UI
    '''
    # todo:: match current skeleton
    # 1. current skeleton 2 screen_name
    # 2. api in the screen
    # 3. use xpaths to judge the api whether it is in the screen
    # 4. show the apis
    ui_apis = {}
    if len(self.raw_log['records'])==0:
      return []
    ui_state = self.raw_log['records'][-1]['State']
    for api_name, api_xpath in self.doc.api_xpath.items():

      root = etree.fromstring(ui_state)
      eles = root.xpath(api_xpath)
      if not eles:
        continue
      ele_desc = etree.tostring(eles[0], pretty_print=True).decode(
          'utf-8')  # only for father node
      id_str = re.search(r' id="(\d+)"', ele_desc).group(1)

      # if '_list' in api_name:
      #     str_elements = '\n\t'.join([etree.tostring(elem, pretty_print=True).decode('utf-8') for elem in eles])
      #     api_desc = f'{api_name}: \n\t\t{str_elements}'
      # else:
      api_desc = api_name
      id = int(id_str)
      ui_apis[id] = api_desc

    # iterate over ui_apis to get the order of apis
    ui_apis_ordered = []
    for id in sorted(ui_apis.keys()):
      ui_apis_ordered.append(ui_apis[id])

    return ui_apis_ordered

  def _get_ordered_ui_apis(self, ui_record):
    ui_apis = {}
    ui_state = ui_record['State']
    for api_name, api_xpath in self.api_xpaths.items():

      root = etree.fromstring(ui_state)
      eles = root.xpath(api_xpath)
      if not eles:
        continue
      ele_desc = etree.tostring(eles[0], pretty_print=True).decode(
          'utf-8')  # only for father node
      id_str = re.search(r' id="(\d+)"', ele_desc).group(1)

      # if '_list' in api_name:
      #     str_elements = '\n\t'.join([etree.tostring(elem, pretty_print=True).decode('utf-8') for elem in eles])
      #     api_desc = f'{api_name}: \n\t\t{str_elements}'
      # else:
      api_desc = api_name
      id = int(id_str)
      ui_apis[id] = api_desc

    # iterate over ui_apis to get the order of apis
    ui_apis_ordered = []
    for id in sorted(ui_apis.keys()):
      ui_apis_ordered.append(ui_apis[id])

    return ui_apis_ordered, ui_state

  def _get_comments_of_all_steps(self):
    records = self.raw_log['records']
    # reorganized_records = []
    # log_str = ''
    script_comments = {}

    for record_id, ui_record in enumerate(records):
      # TODO: this is specific to the notes app, please modify it accordingly
      if (record_id == 0 and ui_record['Action'] == 'go back') or ( # todo::
          record_id == 1 and ui_record['Action'] == 'scroll DOWN'):
        continue

      ui_apis_ordered, ui_state = self._get_ordered_ui_apis(ui_record)

      if 'currently_executing_code' in ui_record.keys():
        if ui_record['currently_executing_code'] is None:
          continue
        code_info = ui_record['currently_executing_code']
        try:
          original_lineno = code_info['original_lineno']
        except:
          print('error info is wrong')
          continue
        script_comments[original_lineno] = ui_apis_ordered
    return script_comments

  def get_commented_script(self):

    def _get_formatted_comment(comment):
      return f'# {{Available UI elements: {comment}}}'

    script_comments = self._get_comments_of_all_steps()
    code = self.raw_solution
    code_lines = code.split('\n')

    code_dict = {}
    for i, line in enumerate(code_lines):
      code_dict[i] = line

    for lineno, comment in script_comments.items():
      leading_spaces = tools.get_leading_tabs(code_dict[lineno])
      code_dict[
          lineno] = f'{leading_spaces}{_get_formatted_comment(comment)}\n{code_dict[lineno]}\n'

    commented_code = '\n'.join([code_dict[i] for i in range(len(code_dict))])
    # commented_code = tools.get_code_without_prefix('tmp/preparation/notes.txt', # todo:: app_name
    #                                                commented_code)
    return commented_code

  def format_all_apis(self, enable_dependency):
    all_elements_desc = ''
    for element in self.ele_data['elements']:
      all_elements_desc += f"\n\nelement: {element['api_name']} \n\tDescription: {element['description']} \n\tType: {element['element_type']}"
      if 'effect' in element.keys():
        all_elements_desc += f"\n\tEffect: {element['effect']}"
    return all_elements_desc

  def make_prompt(self, enable_dependency=False, stuck_ui_apis=None):

    if 'verifier' in self.error_log['error']:
      error_info = 'The statement is not supported by the verifier, please find another way to implement the same functionality.'
    else:
      error_info = self.error_log['error']

    if not stuck_ui_apis:
      stuck_ui_apis = self._get_ui_elements_of_stuck_ui()
      stuck_ui_apis = '\n\t'.join(stuck_ui_apis)

    prompt = f'''A {self.app_name} app in smartphone has the following important UI elements:

{self.format_all_apis(enable_dependency)}

You will be asked to complete tasks by writing scripts to manipulate the above elements.
In the script, except for the common python control flow (for, if-else, function def/calls, etc.), you can use the following APIs:
- tap(<element_selector>): tap on the element. Almost all elements can be taped. If an element's attribute checked=false or selected=false, tapping it can make it checked or selected, vice versa.
- long_tap(<element_selector>): long tap the element. 
- set_text(<element_selector>, <text>): set the text of the element to <text>. Only editable text fields can be set text.
- scroll(<element_selector>, <direction>): scroll the UI element in the specified direction, and direction is a str from "up", 'down", "left", "right". e.g. scroll($scroll_settings_page, "down")
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

Now I give you a task, the current UI state, the former script you have executed before which leads to the current UI state and could not execute any more, and the error message. Based on the information provided, you should return the python script to complete the task. 

The task is:
    {self.task}

Current UI has the following elements:
\t{stuck_ui_apis}

Former script (with UI states recorded at execution) that raises a bug:
```
{self.get_commented_script()}
```

The bug of the former script: 
{error_info}

The script line that caused the error: 
{self.error_log['error_line_in_original_script']}


Your answer should follow this JSON format:

{{
    'Plan': '<the plan to complete the task from the current UI>',
    'Script': '<the Python script to complete the task>',
}}

**Note that the script is a string of python code and should only output the JSON content.**'''

    return prompt

  def process_bug(self,
                  prompt_answer_path,
                  enable_dependency=False,
                  model_name='gpt-3.5-turbo',
                  stuck_ui_apis=None):
    prompt = self.make_prompt(enable_dependency=enable_dependency,
                              stuck_ui_apis=stuck_ui_apis)
    answer = tools.query_gpt(prompt=prompt, model=model_name)
    tools.dump_json_file(prompt_answer_path, {
        'prompt': prompt,
        'answer': answer
    })
    answer = tools.convert_gpt_answer_to_json(answer,
                                              model_name=model_name,
                                              default_value={
                                                  'Plan': '',
                                                  'Script': ''
                                              })
    if 'Script' in answer.keys():
      return answer['Script']
    else:
      return answer['script']


class BugProcessorV3:

  def __init__(self, app_name: str, task: str, doc: ApiDoc, log_path: str, error_path: str, code: str):
    self.app_name = app_name
    self.task = task
    self.doc = doc
    self.raw_log = tools.load_yaml_file(log_path)
    self.error_log = tools.load_json_file(error_path)
    self.code = code

  def get_script_embedded_error(self):
    if 'is not defined' in self.error_log['error']:
      error_info = 'totally wrong API name or variable name'
      return None
    
    if 'verifier' in self.error_log['error']:
      error_info = 'The statement is not supported by the verifier, please find another way to implement the same functionality.'
    else:
      error_info = self.error_log['error']
    
    error_lineno = self.error_log['error_line_number_in_original_script']
    
    code_lines = self.code.split('\n')
    error_line = tools.get_leading_tabs(code_lines[error_lineno]) + f'^^^^^^ {error_info}'
    code_lines.insert(error_lineno + 1, error_line)
    return '\n'.join(code_lines)

  def make_prompt(self, env: environment.AsyncEnv):
    # all elements
    all_elements_desc = self.doc.get_all_element_desc(is_show_xpath=True)

    # current screen elements
    state = env.get_state()
    current_screen_desc = self.doc.get_current_element_desc(state, is_show_xpath=True)
    
    # original script with error
    original_script_with_error = self.get_script_embedded_error()

    instruction = f'''You are a robot operating a smartphone to use the {self.app_name} app. Like how humans operate the smartphone, you can tap, long tap, input text, scroll, and get attributes of the UI elements in the {self.app_name} app. However, unlike humans, you cannot see the screen or interact with the physical buttons on the smartphone. Therefore, you need to write scripts to manipulate the UI elements in the app.'''
    task = f'**Your ultimate task is: {self.task}**'
    use_api = '''\
In the script, except for the common python control flow (for, if-else, function def/calls, etc.), you can use the following APIs:
- tap(<element_selector>) -> None: tap on the element. Almost all elements can be taped. If an element's attribute checked=false or selected=false, tapping it can make it checked or selected, vice versa.
- long_tap(<element_selector>) -> None: long tap the element. 
- set_text(<element_selector>, <text>) -> None: set the text of the element to <text>. Only editable text fields can be set text.
- scroll(<element_selector>, <direction>) -> bool: scroll the UI element in the specified direction, and direction is a str from "up", 'down", "left", "right". e.g. scroll($scroll_settings_page, "down")
- get_text(<element_selector>) -> str: return the text of the element as a string.
- get_attributes(<element_selector>) -> dict[str, str]: return the attributes of the element as a dict, dict keys include "selected", "checked", "scrollable", dict values are boolean. eg. get_attributes($files[3])["selected"].
- back(): close the current window


The <element_selector> primitive is used to select an element, possible ways of selection include:
- $<element id>, eg. $settings_button
- <element_list>[<idx>]: the idx-th in the element list. eg. $my_items[1]

The <element_list> primitive is used to select a list of elements, possible ways of selection include:
- <element_selector>: the items in the list element identified by <element_selector>. eg. $my_items
- <element_list>.match(<text or attribute dict>): the elements in the element list that match the given text or attribute dict. eg. $my_items.match("key words") or $my_items.match({{"selected": true}})
You can use len(<element_list>) to get the total number of items in an element list.

Each <element_selector> can refer to a single element or an element contained multiple elements, especially in the case of complex items within an <element_list>. The following APIs are supported to be invoked as member functions to limit their effect domain: `tap`, `long_tap`, `set_text`, `scroll`, `get_text`, `get_attributes`, and `back`. Note that these APIs still need to satisfy the required arguments. If the APIs are invoked as member functions, they will only affect the element selected by the <element_selector>, while the APIs invoked as global functions will affect all elements in the phone screen. For example, `$note_list[1].tap($note_title)` will tap the title of the second note in the note list, whereas `tap($note_title)` will always tap the first note title in the note list.'''
    original_script_prompt = f"""\
Now, here is an unsuccessful script that you have executed before, where the bug possibly exists in missed <element_selector> due to the inexact element name or failed to invoke API statements because of the incorrect executing order or unexpected results. You should try to fix the bug and re-generate the script to complete the task.

The unsuccessful script with error information is as follows:
```python
{original_script_with_error}
```""" if original_script_with_error else '''\
Here is an example script to complete the task:

```python
# task: Open a note or create a note titled 'note_test' if there is none.
notes = $open_note_title_list

for i in range(len(notes)):
  note = notes[i]
  title = note.get_text($note_title)
  if title == 'note_test':
    note.tap(title)
    return

back()
tap($create_note)
set_text($add_note_title, 'note_test')
tap($text_note_type)
tap($add_note_ok)
```'''
    regenerate_script = f'''\
The unsuccessful execution of the script has changed the screen of the {self.app_name} app. **Therefore, you should generate new script based on the current screen that has the following current UI elements:

{current_screen_desc}

You can use the following important UI elements:

{all_elements_desc}
'''
    output_format = '''Your answer should follow this JSON format:

{
    "plan": "<a high level plan to complete the task>",
    "elements": "<analyze the elements that could be used to complete the task>", 
    "script": "<the python script to complete the task>"
}

**Note that you should only output the JSON content.**'''
    prompt = instruction + '\n' + task + '\n' + original_script_prompt + '\n' + regenerate_script + '\n' + use_api + '\n' + output_format
    return prompt

  def get_solution(self,
                  env: environment.AsyncEnv,
                  prompt_answer_path,
                  model_name='gpt-3.5-turbo'):
    prompt = self.make_prompt(env)
    answer, _ = tools.query_gpt(prompt=prompt, model=model_name)
    tools.dump_json_file(prompt_answer_path, {
        'prompt': prompt,
        'answer': answer
    })
    answer = tools.convert_gpt_answer_to_json(answer,
                                              model_name=model_name,
                                              default_value={
                                                  'Plan': '',
                                                  'Script': ''
                                              })
    if 'Script' in answer.keys():
      return answer['Script']
    else:
      return answer['script']

class BugProcessorV3:

  def __init__(self, app_name: str, task: str, doc: ApiDoc, error_info: dict, code: str, previous_plan:str, err: Exception):
    self.app_name = app_name
    self.task = task
    self.doc = doc
    self.error_log = error_info
    self.code = code
    self.previous_plan = previous_plan
    self.err = err

  def _get_view_without_id(self, view):
    modified_view = re.sub(r" id='\d+'", '', view)
    return modified_view
  
  def get_script_err_prompt(self):
    if isinstance(self.err, (XPathError, APIError, ActionError, NotFoundError)):
      error_info = self.err.msg
    else: # give the example script
      error_info = ",".join(map(str,self.err.args))
    
    error_lineno = self.error_log['error_line_number_in_original_script']
    error_line_info = ""
    code_lines = self.code.split('\n')
    if error_lineno != None:
      error_line_info = f"on line {error_lineno}"
      error_line = tools.get_leading_tabs(code_lines[error_lineno]) + f'^^^^^^ {error_info}'
      code_lines.insert(error_lineno + 1, error_line)
      code_with_err =  '\n'.join(code_lines)
    else:
      code_with_err = f"Error: {error_info} \n\n" + "After executing:" + '\n'.join(code_lines)

    return f"""\
### Previous Attempt: You previously tried to complete the task with the following plan and script:
- Plan: {self.previous_plan}
- Script:
```python
{self.code}
```

**However, the plan didnt work because the script failed to complete the task. An error occurred {error_line_info} in the following script while trying to complete the task:**
- The error message is:
```python
{code_with_err}
```

You can assume that all actions that happened before the error occurred were executed correctly, however, your newly generated script must start again from the begining!
"""

  def make_suggestion(self):
    suggestion = ''
    if isinstance(self.err, APIError):
      suggestion += '**Suggestion:** The error may be caused because of missed <element_selector> due to usage of incorrect element name. Please check the API name in provided UI Elements and the according UI Screen. Select only API elements that are available in the according screen.'
    elif isinstance(self.err, ActionError):
      suggestion += '**Suggestion:** The error may be caused because of failure during invoking API statements because of the incorrect executing order or unexpected results. Please check the rule of the API invocation.'
    elif isinstance(self.err, XPathError):
      suggestion += '**Suggestion:** The error may be caused because of trying to access wrong target element in of current screen. Please check the script logic.'
    elif isinstance(self.err, NotFoundError):
      suggestion += '**Suggestion:** The error may be caused because of trying to access element in the current screen that is not available. Please check the order of invoking APIs and the script logic.'
    else:
      suggestion += '**Suggestion:** The error is internal and may be caused because of incorrect script logic and grammar. Please check the script logic and grammar.'
    suggestion = suggestion + f"\nThis might happen when the execution sequence of the UI elements interactions is incorrect. For example, the previously executed action might lead to a UI screen in which the next chosen api element does not exist. Please revise the script logic!"
    return suggestion
      
  def make_prompt(self, env: environment.AsyncEnv):
    # all elements
    all_elements_desc = self.doc.get_all_element_desc(is_show_xpath=False)
    # current screen
    state = env.get_state()
    # current screen elements
    current_screen_desc = self.doc.get_current_element_desc(state, is_show_xpath=False)

    element_tree = state.element_tree
    visible_html_view = element_tree.get_str_with_visible()
    
    prompt = f'''Imagine that you are a robot operating a smartphone to use the {self.app_name} app. Like how humans operate the smartphone, you can tap, long tap, input text, scroll, and get attributes of the UI elements in the {self.app_name} app. However, unlike humans, you cannot see the screen or interact with the physical buttons on the smartphone. Therefore, you need to write scripts to manipulate the UI elements (buttons, text fields, scrollers, element_lists, etc) in the app.
 
**Your ultimate task is: {self.task}**

{self.get_script_err_prompt()}

### This error occurred on the current screen of the app, which contains the following UI elements and UI APIs accordingly:
- UI Screen State XML:
{visible_html_view}
- UI Screen APIs:
{current_screen_desc}

### You need to fix the error and complete the task by following these steps:
1. Analyze the Error: Why is the script failing? Consider the error message and the UI elements provided and don't make the same mistake!

{self.make_suggestion()}

2. Regenerate a Valid Script: The new script should correct the error, follow the correct grammar of the APIs, and complete the task: {self.task}.
3. Explain Changes: Provide reasoning for why the new script works compared to the previous.

After understanding the problem in the previous script, {ACTIONS_DSL_PROMPT_DESCRIPTION}

You can use the following important UI elements:
{all_elements_desc}


Your answer should follow this JSON format:
{{
    "error_analysis": "<string, analysis of the error in the previous script; understanding what is the problem and why it failed>",
    "plan": "<string, revised high level plan to complete the task; addressing the problems in the previous plan>",
    "elements": "<string, analysis on the elements that could be used to complete the task; why each element is suitable for this task>", 
    "script_changes": "<string, compare the changes of the new and the previous script and explain why the new script is better>",
    "script": "<string, provide a new different script that is responsible for completing the task>",
}}


**Note that you should only output the JSON content.**'''
    return prompt

  def get_fixed_solution(self,
                  env: environment.AsyncEnv,
                  prompt_answer_path,
                  model_name='gpt-3.5-turbo'):
    prompt = self.make_prompt(env)
    answer, tokens = tools.query_gpt(prompt=prompt, model=model_name)
    answer, tokens1 = tools.convert_gpt_answer_to_json(answer,
                                              model_name=model_name,
                                              default_value={
                                                  'Plan': '',
                                                  'Script': ''
                                              })
    tools.append_to_txt_file(prompt_answer_path, f'Prompt: {prompt}\n'+'='*50+'\nAnswer: {answer}\n'+'*'*50+'\n')
    tools.dump_json_file(prompt_answer_path, {
        'prompt': prompt,
        'answer': answer,
        'tokens': tokens,
        'convert_tokens': tokens1
    })
    if 'Script' in answer.keys():
      return answer['Script'], answer['Plan']
    else:
      return answer['script'],  answer['plan']
    
  def fix_invalid_xpath(self,
                      env: environment.AsyncEnv, 
                      api_name: str,
                      prompt_answer_path,
                      model_name='gpt-3.5-turbo'):
    api = self.doc.get_api_by_name(api_name)
    if not api:
      return
    
    state = env.get_state()
    element_tree = state.element_tree
    
    if self.doc.check_api_name_in_current_screen(api_name, element_tree.skeleton):
      # navigating is already changed to another screen
      # todo:: only first miss match to trigger the navigating
      return False
    
    element_tree_html_without_id = self._get_view_without_id(element_tree.str)
    
    element_html = api.element
    element_html_without_id = self._get_view_without_id(element_html)
    prompt = f'''\
Given the current UI state, you should find the valid XPath for the UI element that is represented by the following HTML code:

{element_html_without_id}

The current UI state is described by the following HTML code:

{element_tree_html_without_id}

Your answer should follow this JSON format:

{{
  "flag": <bool, whether the target element exists in the current UI state>,
  "xpath": <string, the unique xpath of the target element>
}}

**Note that you should only output the JSON content.**'''
    answer, tokens = tools.query_gpt(prompt=prompt, model=model_name)
    answer, tokens1 = tools.convert_gpt_answer_to_json(answer,
                                              model_name=model_name,
                                              default_value={
                                                  'flag': False,
                                                  'xpath': ''
                                              })
    
    tools.dump_jsonl_file(prompt_answer_path, {
        'prompt': prompt,
        'answer': answer,
        'tokens': tokens,
        'convert_tokens': tokens1
    })
    
    flag = answer.get('flag', None)
    if flag == False:
      return False
    
    xpath = answer.get('xpath', None)
    if xpath == None:
      return False
    
    ele = element_tree.get_ele_by_xpath([xpath])
    if not ele:
      return False
    
    api.xpath = [xpath]
    self.doc.is_updated = True
    return True
