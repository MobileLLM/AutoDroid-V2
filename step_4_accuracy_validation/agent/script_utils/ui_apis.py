import json
import os
import re
import yaml
import inspect
import time
import datetime

import logging
import tools as tools
import agent.agent_utils as agent_utils
import agent.environment as environment

from agent.droidbot.device_state import ElementTree, EleAttr, DeviceState

from agent.script_utils.api_doc import ApiDoc
from agent.script_utils.err import XPathError, APIError, ActionError, NotFoundError

from . import MAX_SCROLL_NUM, MAX_ACTION_COUNT, LOGGING_ENABLED, MAX_DEPENDENCE_DEPTH, MAX_DEPENDENCE_WIDTH, WAIT_AFTER_ACTION_SECONDS

api_names = [
    'long_tap', 'tap', 'set_text', 'scroll', 'get_text', 'get_attributes',
    'back', 'get_ui_tree', 'check_ele_exist', 'enter'
]

def sanitize_name(name):
  # To make it a valid python variable, replace all non-word characters with '_', and replace the first digit with '_'
  return re.sub(r'\W|^(?=\d)', '_', name)

  
def regenerate_script(script, verifier_instant_name):
  '''
    find element_lists and instantiate them, remove '$' from element_selectors, add instant_name prefix to all apis
    '''
  pattern = re.compile(r'^.*?\$([\w%]+).*?(\[\d+\]|\.match\([^)]+\)).*$',
                       re.MULTILINE)
  script_lines = script.split('\n')
  modified_lines = [
      f'def autodroidv2_task_solution_code({verifier_instant_name}):'
  ]  # def a function because of the necessity of inspecting the script
  all_appeared_api_names = []
  line_mappings = {}  # key: compiled script line number, value: original script line number
  element_statement_set = set()
  
  for _, line in enumerate(script_lines):
    match = pattern.match(line)
    if match:
      # for matching, indexing operation statements.
      element_name = match.group(1)
      sanitized_element_name = sanitize_name(element_name)
      line = line.replace(f'${element_name}', sanitized_element_name)
      
      element_statement_set.add(f'{sanitized_element_name} = ElementList(\'{element_name}\', None, {verifier_instant_name})')
    
    # for tapping, set_text, etc. statements
    api_name_pattern = r'\$([\w%]+)'  # also match apis with %, for example, font_size_150%
    matches = re.findall(api_name_pattern, line)
    if matches:
      for api_name in matches:
        sanitized_api_name = sanitize_name(api_name)
        if sanitized_api_name not in all_appeared_api_names:
          all_appeared_api_names.append(api_name)
          element_statement_set.add(f'{sanitized_api_name} = ElementList(\'{api_name}\', None, {verifier_instant_name})')

        line = line.replace(f'${api_name}', sanitized_api_name)
    modified_lines.append(f'\t{line}')
  
  element_statement_list = list(element_statement_set)
  element_statement_list.sort()
  statement_len = len(element_statement_list)
  beginning_tabs = tools.get_leading_tabs(modified_lines[1])
  
  for s in element_statement_list:
    modified_lines.insert(1, beginning_tabs + s)
  
  for i, _ in enumerate(modified_lines[statement_len + 1:]):
    original_line_num = i
    compiled_line_num = i + statement_len + 1
    line_mappings[compiled_line_num] = original_line_num

  modified_lines.append(
      f'autodroidv2_task_solution_code({verifier_instant_name})'
  )
  script = '\n'.join(modified_lines)
  script = script.replace('```python', '').replace('```', '').replace('python', '').replace('Python', '')
  script = script.replace('back()', f'{verifier_instant_name}.back()')
  script = script.replace('enter()', f'{verifier_instant_name}.enter()')

  # for api_name in api_names:
  #   script = script.replace(f'{api_name}(',
  #                           f'{verifier_instant_name}.{api_name}(')
  #   script = script.replace(f'.{verifier_instant_name}.{api_name}(', f'.{api_name}(')
  # script = script.replace(f'long_{verifier_instant_name}.tap(', 'long_tap(')
  return script, line_mappings


def _save2yaml(file_name,
               state_prompt,
               idx,
               inputs=None,
               action_type='touch',
               api_name=None,
               xpath=None,
               skeleton=None,
               tag=None,
               raw_prompt=None,
               raw_answer=None,
               currently_executing_code=None,
               target='action',
               effect_range='global',
               screenshot=None):
  if not LOGGING_ENABLED:
    return
  
  if not os.path.exists(file_name):
    tmp_data = {'step_num': 0, 'records': []}
    with open(file_name, 'w', encoding='utf-8') as f:
      yaml.dump(tmp_data, f)

  with open(file_name, 'r', encoding='utf-8') as f:
    old_yaml_data = yaml.safe_load(f)
  new_records = old_yaml_data['records']
  new_records.append({
      'step': len(new_records),
      'State': state_prompt,
      'Choice': idx,
      'Action': action_type,
      'Input': inputs,
      'api_name': api_name,
      'xpath': xpath,
      'skeleton': skeleton,
      'tag': tag,
      'target': target,
      'raw_prompt': raw_prompt,
      'raw_answer': raw_answer,
      'currently_executing_code': currently_executing_code,
      'effect_range': effect_range,
      'screenshot': screenshot})
  data = {
      'step_num': len(new_records),
      'records': new_records
  }
  t1 = time.time()
  with open(file_name, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data, f)
  print(f'save to yaml time: {time.time() - t1}')

def _save2log(save_path, 
               log_file: str,
               element_tree: ElementTree = None,
               idx=None,
               inputs=None,
               action_type='touch',
               api_name=None,
               xpath=None,
               currently_executing_code=None,
               comment: str = 'action',
               effect_range: str = 'global',
               screenshot = None):
  if not LOGGING_ENABLED:
    return
  
  timestamp = datetime.datetime.now().strftime('%Y-%m-%d_T%H%M%S')
  _save2yaml(
    file_name=log_file,
    state_prompt=element_tree.str if element_tree else None,
    idx=idx,
    inputs=inputs,
    action_type=action_type,
    api_name=api_name,
    xpath=xpath,
    skeleton=element_tree.skeleton.str if element_tree else None,
    tag=timestamp,
    raw_prompt=None,
    raw_answer=None,
    currently_executing_code=currently_executing_code,
    target=comment,
    effect_range=effect_range,
    screenshot=screenshot
  )


class CodeConfig:
  def __init__(self, 
               app_name: str, 
               doc: ApiDoc):
    self.app_name = app_name
    self.doc = doc
    
    self.save_path = None
    self.log_file = None
    self.code: str = None
    self.compiled_code: str = None
    self.line_mappings: dict[int, int] = None
    
    self.code_lines: list[str] = None
    self.compiled_code_lines: list[str] = None

  def set(self, save_path: str, code: str, compiled_code: str, line_mappings: dict[int, int]):
    self.save_path = save_path
    self.log_file = os.path.join(save_path, 'log.yaml')
    
    self.code = code
    self.compiled_code = compiled_code
    self.line_mappings = line_mappings
    
    self.code_lines = code.split('\n')
    self.compiled_code_lines = compiled_code.split('\n')
    
    self.enable_dependency = False


class CodeStatus:
  def __init__(self):
    # internal
    self.action_count = 0
    self.last_screen_html_str = None
    
  def reset(self):
    self.action_count = 0
    self.last_screen_html_str = None
    
  def check_action_count(self):
    if self.action_count >= MAX_ACTION_COUNT:
      raise Exception(f'Action count is over {MAX_ACTION_COUNT}, the script may be in an infinite loop')
      # pass
    self.action_count += 1
  
  def check_last_screen(self, screen_html_str: str):
    is_same = False
    if not self.last_screen_html_str:
      self.last_screen_html_str = screen_html_str
    else:
      is_same = self.last_screen_html_str == screen_html_str
      self.last_screen_html_str = screen_html_str
    return is_same


class Verifier:

  def __init__(self, env: environment.AsyncEnv, config: CodeConfig, status: CodeStatus) -> None:
    # android world
    self.env = env
    self.save_path: str = config.save_path
    self.app_name: str = config.app_name
    self.doc: ApiDoc = config.doc
    self.api_xpaths = self.doc.api_xpath
    self.config = config
    
    # status
    self.status = status
    
    self._state = None
    self._element_tree = None

    self.gpt_located_elements = []

  @property
  def state(self):
    return self.env.get_state()

  @property
  def element_tree(self):
    self._element_tree = self.state.element_tree
    return self._element_tree
  
  @property
  def action_count(self):
    return self.status.action_count

  def check_action_count(self):
    self.status.check_action_count()
  
  @property
  def last_screen(self):
    return self.status.last_screen_html_str
  
  def get_cached_element_tree(self):
    if self._element_tree:
      return self._element_tree
    return self.element_tree
  
  def get_fresh_state(self):
    self.env.wait_for_stable_state(2,0)
    return self.state
  
  def update_state(self):
    self.env.wait_for_stable_state()

  def check_last_screen_html(self):
    is_same = self.status.check_last_screen(self.element_tree.str)
    return is_same

  def get_unique_xpath_on_screen(self, api_name, xpaths):
    if isinstance(api_name,list):
      api_name = api_name[0]
    screen = api_name.split('__')[0]
    unique = []
    for xpath in xpaths:
      is_duplicate = False
      for el in self.doc.doc[screen].values():
        if el.api_name == api_name:
          continue
        if xpath in el.xpath:
          is_duplicate = True
          break
      if not is_duplicate:
        unique.append(xpath)
    return unique
  
  def locate_element(self, api_name, xpaths, use_gpt=False):
    if len(xpaths) > 0 or not use_gpt:
      found = None
      for path in xpaths:
        found = self.element_tree.get_ele_by_xpath(path)
        if found:
          break
      return found
    else:
      return self.query_model_for_locating_element(api_name)
    
    
  def scroll_and_find_target_ele(self,
                                 api_name,
                                 xpaths,
                                 statement,
                                 direction='DOWN'):
    
    scrolling_direction = direction.lower()
    target_ele = self.locate_element(api_name, xpaths)
    if target_ele != None:
      return target_ele
    
    for ele_id in self.element_tree.scrollable_ele_ids:
      scrollable_element = self.element_tree.ele_map[ele_id]
    # scrollable_element = self.element_tree.ele_map[self.element_tree.root.id]
      WAYS_OF_ELEMENT_LOCATION = ['xpath', 'gpt']
      for locator in WAYS_OF_ELEMENT_LOCATION:
        use_gpt = locator == 'gpt'
        # change direction and try again
        if use_gpt:
          scrolling_direction = "up"
        
        for _ in range(MAX_SCROLL_NUM):
          
          target_ele = self.locate_element(api_name, xpaths, use_gpt=use_gpt)
          if target_ele != None:
            return target_ele
        
          _save2log(
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=self.element_tree,
            idx=scrollable_element.id if scrollable_element else None,
            inputs=None,
            action_type=f'scroll {scrolling_direction}',
            api_name=None,
            xpath=xpaths,
            currently_executing_code=statement,
            comment='navigate',
            screenshot=self.state.screenshot)

          self.env.execute_action(
              {
                  "action_type": "scroll",
                  "view": scrollable_element.view,
                  "direction": scrolling_direction
              })
          time.sleep(WAIT_AFTER_ACTION_SECONDS)
          self.update_state()

          is_same = self.check_last_screen_html()
          if is_same:
            break
            

    return None

  def query_model_for_locating_element(self, api_name):
    state_xml = self.element_tree.str
    api_screen = self.doc.get_api_screen_name(api_name)
    el = self.doc.doc[api_screen][api_name]
    # def encode_image(image_path):
    #   with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode('utf-8')
        
    # state_image = encode_image(self.state.screenshot)
    # prompt = []
    # prompt.append({
    #           "type": "image_url",
    #           "image_url": {
    #               "url": f"data:image/jpeg;base64,{state_image}",
    #           }
    #       })
    prompt_text = f''''
# task: Locate the api: '${api_name}' in the current UI screen of a mobile application by selecting the UI element that corresponds the best to the api name. 

- The purpose of this api is: {el.description}.{el.effect}.
- The possible xpath that leads to the element is: {el.xpath[0]}
- The screen is represented in xml format as follows:
{state_xml}

# example: Return the element notation of the api_name element, including all element's attributes, excluding the children elements.
- Example api name: $settings_screen__privacy_setting_button
- Example output: {{
  "element": "<button id='11' resource-id='security_settings' alt='Privacy and Security'>"
}}

# reasoning: Think about your reasoning for selecting the element step by step. What makes you think this element corresponds to the api name? What are the key attributes of the element that match the api name? 
- When the api name is related to a 'list', the desired element should be an abstraction element (layout element, scrollbar, list element, etc.) that contains a list of similar children.

# output: The final answer must be in the following json format:
```
{{
  "element": element_notation | "None"
}}
```
Please output only answer that is in the format of the json above and nothing else!
    '''
    # prompt.insert(0,{
    #         "type": "text",
    #         "text": prompt_text,
    # })
    
    t1 = time.time()
    answer, tokens = tools.query_gpt(prompt=prompt_text, model="gpt-4")
    # _save2log(
    #       save_path=self.save_path,
    #       log_file=self.config.log_file,
    #       element_tree=element_tree,
    #       idx=None,
    #       inputs=None,
    #       action_type=None,
    #       api_name=api_name,
    #       xpath=el.xpath,
    #       currently_executing_code=statement,
    #       comment='crashed',
    #       screenshot=state.screenshot)
    print("GPT Located Element: ", answer)
    json_answer = json.loads(answer.replace("```",'').replace('json',''))
    t2 = time.time()
    element = json_answer["element"]
    self.gpt_located_elements.append({
      "api_name": api_name,
      "located": element,
      'prompt': prompt_text,
      'answer': answer,
      'tokens': tokens,
      "time": t2 - t1
    })

    # print(f"Found element: {element}")
    # time.sleep(10)
    if element == 'None':
      return None
    else:
      element_id = int(re.search(r"id='(\d+)'", element).group(1))
      return self.element_tree.get_ele_by_id(element_id)

  def _track_element_by_dependencies(self, api_name, xpath, statement):
    counter = 0
    target_ele = None
    while target_ele is None and counter < MAX_DEPENDENCE_WIDTH:
      _, dependency_action = self.doc.get_dependency(api_name)
      
      if not dependency_action:
        break
      
      count = 0
      for action_list in dependency_action[:MAX_DEPENDENCE_DEPTH]:
        count += 1

        is_match = False
        dep_id = -1
        for idx, action in enumerate(reversed(action_list)):
          element_tree = self.element_tree
          
          # try to find the target element in the current UI
          target_ele = element_tree.get_ele_by_xpath(xpath)
          if target_ele != None:
            break
          
          current_screen_name = self.doc.get_screen_name_by_skeleton(element_tree.skeleton)
          if action.screen_name != current_screen_name:
            continue
          
          if action.action_type == 'back' or action.action_type == 'enter':
            is_match = True
            dep_id = idx
            _save2log(
              save_path=self.save_path,
              log_file=self.config.log_file,
              element_tree=element_tree,
              idx=None,
              inputs=None,
              action_type=action.action_type,
              api_name=None,
              xpath=None,
              currently_executing_code=statement,
              comment='device action',
              screenshot=self.state.screenshot)
            self.env.execute_action(
                {
                    "action_type": action.action_type
                })
            # time.sleep(WAIT_AFTER_ACTION_SECONDS)
            self.update_state()
            continue
          
          _action_xpath = self.doc.api_xpath.get(action.name, None)
          if not _action_xpath:
            continue
          _target_ele = self.find_and_scroll_target_ele(_action_xpath, statement)
          
          if not _target_ele:
            if is_match:
              break
            else:
              continue

          # execute the action
          is_match = True
          dep_id = idx
          _save2log(
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=element_tree,
            idx=_target_ele.id if _target_ele else None,
            inputs=None,
            action_type=action.action_type,
            api_name=action.api_name,
            xpath=_action_xpath,
            currently_executing_code=statement,
            comment='navigate',
            screenshot=self.state.screenshot)
          
          executable_action = agent_utils.convert_action(action.action_type, _target_ele, action.text)
          # finding dependency can tolerate the action error
          # if executable_action.get('action_type') == 'wait':
          #   raise ActionError(f'Fail to {action.action_type}({action.api_name})', None, None, action.action_type, action.api_name)
          self.env.execute_action(executable_action)
          time.sleep(WAIT_AFTER_ACTION_SECONDS)
          self.update_state()
          self.check_last_screen_html()

        if dep_id >= len(action_list) - 1:
          element_tree = self.element_tree
          target_ele = element_tree.get_ele_by_xpath(xpath)              
          break
          # if target_ele is None, continue to find the next dependency
        
        # executed action and changed the screen, we need to find new dependency
        if is_match:
          break

      if count == len(dependency_action):
        # fail to solve the dependency
        # target_ele still is None
        break
      
      counter += 1
    return target_ele
  def get_and_navigate_target_element(self, api_name, xpath, statement):
    print(f"--- Looking for api_name: {api_name} ---")
    print(f"--- Using xpath: {xpath} ---")
    print("--- Current UI Tree ---")
    try:
        print(self.element_tree.str)
    except Exception as e:
        print(f"Could not print element tree: {e}")
    print("--- End of UI Tree ---")
    # print(f"Looking for xpaths:{xpath}")
    # print(self.element_tree.str)
    # time.sleep(10)
    t1 = time.time()
    if api_name == None:
      #  This case happens only when the following code is executed: element_list.match("").tap()
      #  on the action tap(), the api_name is None because maybe the matched element is not the document,
      #  but it is sure that the xpath the matched element is set, hence we can directy locate it.
      target_ele = self.element_tree.get_ele_by_xpath(xpath)
    else:
      unique_xpaths = self.get_unique_xpath_on_screen(api_name, xpath)
      # print(f"Looking for unique xpaths:{unique_xpaths}")
      target_ele = self.scroll_and_find_target_ele(api_name, unique_xpaths, statement)
    
    # could not find a target element in the current UI, find in the dependencies
    if target_ele == None:
      is_in_current_screen = self.doc.check_api_name_in_current_screen(api_name, self.element_tree.skeleton)
      if is_in_current_screen:
        # assume the target element is in the current screen
        # but we still can't find it, so raise an error
        raise XPathError(f'Not Exist {api_name}[{xpath}]', api_name, xpath)
      # else:
      # not in screen, try to navigate to the screen
      
      if not self.config.enable_dependency:
        raise NotFoundError(f'Not found {api_name}[{xpath}]', api_name, xpath)
      
      ## navigating in dependency
      # we have executed all the dependencies, but still not found the target element
      target_ele = self._track_element_by_dependencies(api_name, xpath, statement)
      
    if target_ele != None:
      t2 = time.time()
      time_spend_locating = t2 - t1
      return target_ele, time_spend_locating
    else:
      _save2log(
          save_path=self.save_path,
          log_file=self.config.log_file,
          element_tree=self.element_tree,
          idx=None,
          inputs=None,
          action_type=None,
          api_name=api_name,
          xpath=xpath,
          currently_executing_code=statement,
          comment='crashed',
          screenshot=self.state.screenshot)
      raise NotFoundError(f'Not found {api_name}[{xpath}]', api_name, xpath)
  
  def check_api(self, api, action_type, statement, text=None):
    if isinstance(api, str):
      button_api_name = api.split('$')[-1]
      api_name = button_api_name
      try:
        xpath = self.api_xpaths[button_api_name]
      except KeyError:
        _save2log( # save crash in get_and_navigate_target_element
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=self.element_tree,
            idx=None,
            inputs=text,
            action_type=action_type,
            api_name=api_name,
            xpath=None,
            currently_executing_code=statement,
            comment='crashed',
            screenshot=self.state.screenshot)
        raise APIError(f'Invalid {button_api_name}', button_api_name)
    else:
      api_name = api.api_name,
      xpath = api.element_list_xpath
    return api_name, xpath
  
  def _execute_action(self, api_name, xpath, statement, action_type, text: str=None):
    target_ele, time_spend_locating = self.get_and_navigate_target_element(api_name, xpath, statement)
    
    if action_type == 'set_text':
      _, y = DeviceState.get_view_center(target_ele.view)
      width, height = self.env.logical_screen_size
      
      if y >= 0.9 * height:
        _save2log(
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=self.element_tree,
            idx=target_ele.id,
            inputs=text,
            action_type='scroll down',
            api_name=None,
            xpath=None,
            currently_executing_code=statement,
            comment='navigate',
            screenshot=self.state.screenshot)
        self.env.execute_action(
            {
                "action_type": "scroll",
                "view": target_ele.view, # scroll down target element
                "direction": 'down', # it happens nothing when it is not scrollable,
                "time_spent_locating": time_spend_locating
            })
        time.sleep(WAIT_AFTER_ACTION_SECONDS)
        self.update_state()
        element_tree = self.element_tree
        target_ele = element_tree.get_ele_by_xpath(xpath)
      
    _save2log( # save crash in get_and_navigate_target_element
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=text,
        action_type=action_type,
        api_name=api_name,
        xpath=xpath,
        currently_executing_code=statement,
        comment='action',
        screenshot=self.state.screenshot)
    
    executable_action = agent_utils.convert_action(action_type, target_ele, text)
    self.env.execute_action(executable_action)
    time.sleep(WAIT_AFTER_ACTION_SECONDS)
    self.update_state()
    # print(f"action executed {api_name} {target_ele.full_desc}")
    self.check_action_count()
  
  def tap(self, button_api):
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(
        f"Tap: {button_api} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    action_type = 'touch'
    statement = {
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    api_name, xpath = self.check_api(button_api, action_type, statement)
    self._execute_action(api_name, xpath, statement, action_type)

  def long_tap(self, button_api):
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(
        f"long tap: {button_api} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}"
    )
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    action_type = 'touch'
    statement = {
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    api_name, xpath = self.check_api(button_api, action_type, statement)
    self._execute_action(api_name, xpath, statement, action_type)

  def set_text(self, input_api, text):
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"set_text: {input_api} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    action_type = 'set_text'
    statement = {
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    api_name, xpath = self.check_api(input_api, action_type, statement, text)
    self._execute_action(api_name, xpath, statement, action_type, text)

  def scroll(self, scroller_api, direction):
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"scroll {direction}: {scroller_api} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    last_screen = self.last_screen
    
    statement = {
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    if 'up' in direction.lower():
      direction_str = 'up'
    elif 'down' in direction.lower():
      direction_str = 'down'
    elif 'left' in direction.lower():
      direction_str = 'left'
    elif 'right' in direction.lower():
      direction_str = 'right'
    elif 'page_right' in direction.lower():
      direction_str = 'page_right'
    elif 'page_left' in direction.lower():
      direction_str = 'page_left'
    elif 'page_down' in direction.lower():
      direction_str = 'page_down'
    elif 'page_up' in direction.lower():
      direction_str = 'page_up'
    else:
      direction_str = 'down'
    action_type = f'scroll {direction_str}'
    
    api_name, xpath = self.check_api(scroller_api, action_type, statement)
    self._execute_action(api_name, xpath, statement, action_type)
    is_to_bottom = False if not last_screen else self.last_screen == last_screen
    return is_to_bottom

  def get_text(self, element_selector):
    '''
    return the text of the element as a string.
    '''

    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"get_text: {element_selector} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    statement={
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    # for actions like getting length, indexing, or matching, the element_selector is a string
    if isinstance(element_selector, str):
      element_selector = element_selector.split('$')[-1]

      element_selector_xpath = self.api_xpaths[element_selector]
      element_selector_api_name = element_selector
    else:
      if isinstance(element_selector, list):
        element_selector = element_selector[0]
      element_selector_xpath = element_selector.element_list_xpath
      element_selector_api_name = element_selector.api_name if element_selector.api_name else element_selector.element_list_xpath
    
    target_ele, time_spent_locating = self.get_and_navigate_target_element(
        element_selector_api_name,
        element_selector_xpath,
        statement)
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=None,
        action_type='get_text',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None) # same as the current screen
    
    self.check_action_count()
    # not change the status
    
    text = self.element_tree.get_text(target_ele)
    text = text.replace('--', ' ') if text else ''
    return text

  def get_attributes(self, element_selector):
    '''
    return the attributes of the element as a dict, dict keys include "selected", "checked", "scrollable", dict values are boolean. eg. get_attributes($files[3])["selected"].
    '''

    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"get_attributes: {element_selector} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    statement={
        'current_code': current_code_line,
        'original_lineno': lineno_in_original_script,
        'original_code': original_code_line
    }
    # for actions like getting length, indexing, or matching, the element_selector is a string
    if isinstance(element_selector, str):
      element_selector = element_selector.split('$')[-1]

      element_selector_xpath = self.api_xpaths[element_selector]
      element_selector_api_name = element_selector
    else:
      if isinstance(element_selector, list):
        element_selector = element_selector[0]
      element_selector_xpath = element_selector.element_list_xpath
      element_selector_api_name = element_selector.api_name if element_selector.api_name else element_selector.element_list_xpath
    
    target_ele, time_spent_locating = self.get_and_navigate_target_element(
        element_selector_api_name,
        element_selector_xpath,
        statement)
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=None,
        action_type='get_attributes',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None) # same as the current screen
    
    self.check_action_count()
    # not change the screen
    
    target_ele_attrs = target_ele.get_attributes()
    target_ele_attrs['text'] = target_ele_attrs['text'].replace('--', ' ') if target_ele_attrs['text'] else ''
    return target_ele_attrs

  def enter(self):
    '''
    submit a form/input field
    '''

    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"enter at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    self.update_state()

    element_tree = self.element_tree

    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=element_tree,
        idx=None,
        inputs=None,
        action_type='enter',
        api_name=None,
        xpath=None,
        currently_executing_code={
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        },
        screenshot=self.state.screenshot)

    self.env.execute_action({"action_type": "enter"})
    time.sleep(WAIT_AFTER_ACTION_SECONDS)
    
    self.check_action_count()

  def back(self):
    '''
    close the current window
    '''
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    print(f"back at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    self.update_state()

    element_tree = self.element_tree

    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=element_tree,
        idx=None,
        inputs=None,
        action_type='back',
        api_name=None,
        xpath=None,
        currently_executing_code={
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        },
        screenshot=self.state.screenshot)
    self.env.execute_action({"action_type": "navigate_back"})
    time.sleep(WAIT_AFTER_ACTION_SECONDS)
    
    foreground_activity_name = self.env.foreground_activity_name
    # out of the app
    if foreground_activity_name and foreground_activity_name.startswith('com.google.android.apps.nexuslauncher'):
      self.env.execute_action({"action_type": "open_app", "app_name": self.app_name})
      time.sleep(WAIT_AFTER_ACTION_SECONDS)
    self.check_action_count()


class ElementList:

  def __init__(self, api_name, api_xpath, verifier: Verifier) -> None:
    # all element_lists can be uniquely identified by their api_xpath. If one api_name is provided, we can retrieve its xpath from api_xpaths. If api_name is not provided, such as a dynamic element at runtime, then its xpath must be provided.
    self.env = verifier.env
    self.save_path = verifier.save_path
    self.config = verifier.config
    self.doc = verifier.doc
    
    self.api_name = api_name
    self.api_xpaths = verifier.api_xpaths
    
    if self.api_name:
      self.check_api_name(api_name)
    if not api_xpath:
      self.element_list_xpath = self.api_xpaths[api_name]
    else:
      self.element_list_xpath = [api_xpath] # __getitem__
    self.verifier = verifier
    self.index = 0
    
    self.status = verifier.status
  
  @property
  def state(self):
    return self.verifier.state

  @property
  def element_tree(self):
    return self.verifier.element_tree
  
  @property
  def cached_element_tree(self):
    return self.verifier.get_cached_element_tree()
  
  @property
  def action_count(self):
    return self.status.action_count
  
  def check_action_count(self):
    self.status.check_action_count()
  
  def check_last_screen_html(self):
    return self.verifier.check_last_screen_html()
  
  def update_state(self):
    self.verifier.update_state()

  def check_api_name(self, api_name):
    if api_name not in self.api_xpaths.keys():  # not found xpath
      # find the first line with the api_name in the original script (combined with the preparation, this is to stay the same with tap, set_text, etc.)
      raise APIError(f'Invalid {api_name}', api_name)

  def convert_ele_attr_to_elementlist(self, ele_attr):
    ele_xpath = f"//{ele_attr.type_}[@id='{ele_attr.id}']"
    elementlist = ElementList(
        api_name=None,
        api_xpath=ele_xpath,
        verifier=self.verifier)
    return elementlist

  def __getitem__(self, selector):
    # get the currently executing code
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, f'index[{selector}]', selector)
    
    element_selector_api_name = self.api_name if self.api_name else self.element_list_xpath
    element_selector_xpath = self.element_list_xpath
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    target_ele_group, _ = self.verifier.get_and_navigate_target_element(element_selector_api_name, element_selector_xpath, statement)
    
    # Default to integer index if not a custom selector
    if isinstance(selector, int):
      ele_attr = self.element_tree.get_children_by_idx(target_ele_group, selector)
      if ele_attr is None:
        raise Exception(f"Fail to __getitem__({selector}) in {self.api_name}[{self.element_list_xpath}], the index is out of range. This could be because the selector is not a list but the element itself or the index is out of range.")
      matched_ele = self.convert_ele_attr_to_elementlist(
          ele_attr)
      
      _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele_group.id,
        inputs=selector,
        action_type=f'__index__',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None)
      
      self.check_action_count()
      return matched_ele
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele_group.id,
        inputs=selector,
        action_type=f'__index__',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        comment='crashed',
        screenshot=self.state.screenshot)
    raise ActionError(f"Fail to __getitem__({selector}) in {self.api_name}[{self.element_list_xpath}]", self.api_name, self.element_list_xpath, '__getitem__', selector)

  def __iter__(self):
    '''
        in order to support iteration, we need to return an iterator object from __iter__() method.
        '''
    return self

  def __next__(self):
    '''
    return the next element in the current element's children to support iteration.
    '''
    # get the currently executing code
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_back.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, '__next__', self.api_name)
    
    element_selector_api_name = self.api_name if self.api_name else self.element_list_xpath
    element_selector_xpath = self.element_list_xpath
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    target_ele_group, _ = self.verifier.get_and_navigate_target_element(element_selector_api_name, element_selector_xpath, statement)
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele_group.id,
        inputs=self.index,
        action_type='__next__',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None)

    ele_list_children = self.element_tree.get_children_by_ele(target_ele_group)
    if not ele_list_children:
      raise StopIteration
    self.check_action_count()
    if self.index < len(ele_list_children):
      ele_attr = ele_list_children[self.index]
      matched_ele = self.convert_ele_attr_to_elementlist(
          ele_attr)
      self.index += 1
      return matched_ele
    else:
      self.index = 0
      raise StopIteration

  def match(self, match_data):
    # get the currently executing code
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'match', match_data)

    element_selector_api_name = self.api_name if self.api_name else self.element_list_xpath
    element_selector_xpath = self.element_list_xpath
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    target_ele, _ = self.verifier.get_and_navigate_target_element(element_selector_api_name, element_selector_xpath, statement)    
    
    # matched_elements = []
    
    def is_match(ele, match_data):
      if ele == None:
        return False
      if isinstance(match_data, str):
        return ele.is_match(match_data)
      elif isinstance(match_data, dict):
        return ele.is_match_dict(match_data)
      
    def try_find_match(list_ele, match_data):
      found_match = is_match(list_ele, match_data)
      if found_match:
            matched_ele = self.convert_ele_attr_to_elementlist(list_ele)
            return matched_ele
      for child_id in list_ele.children:
          # if child_id in self.cached_element_tree.valid_ele_ids:
          ele = self.cached_element_tree.ele_map.get(child_id, None)
          if ele != None:
            matched = try_find_match(ele, match_data)
            if matched != None:
              return matched
      return None
    
    # def try_find_match(items, match_data):
    #   for ele in items:
    #     if isinstance(match_data, str):
    #       if ele.is_match(match_data):
    #         matched_ele = self.convert_ele_attr_to_elementlist(ele)
    #         matched_elements.append(matched_ele)
    #     elif isinstance(match_data, dict):
    #       ele_dict = ele.dict()
    #       if all(ele_dict[key] == value for key, value in match_data.items()):
    #         matched_ele = self.convert_ele_attr_to_elementlist(ele)
    #         matched_elements.append(matched_ele)
    #   return len(matched_elements) > 0

    found_match = try_find_match(target_ele, match_data)

    # todo:: how to deal with multiple matched elements
    if found_match == None:
      el_tree = self.element_tree
      root_ele = el_tree.ele_map[el_tree.root.id]
      # matched_elements = []
      found_match = try_find_match(root_ele, match_data)
    
    if found_match == None:
      direction = "down"
      scrollable_element = target_ele

      for _ in range(MAX_SCROLL_NUM):
        _save2log(
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=self.element_tree,
            idx=scrollable_element.id,
            inputs=None,
            action_type=f'scroll {direction}',
            api_name=None,
            xpath=element_selector_xpath,
            currently_executing_code=statement,
            comment='navigate',
            screenshot=self.state.screenshot)
        self.env.execute_action(
            {
                "action_type": "scroll",
                "view": scrollable_element.view,
                "direction": direction
            })
        time.sleep(WAIT_AFTER_ACTION_SECONDS)
        self.update_state()

        is_same = self.check_last_screen_html()
        if is_same:
          break

        # ele_list_children = self.element_tree.get_children_by_ele(scrollable_element)
        found_match = try_find_match(scrollable_element, match_data)
        if found_match != None:
          break

    if found_match == None:
      _save2log(
          save_path=self.save_path,
          log_file=self.config.log_file,
          element_tree=self.element_tree,
          idx=target_ele.id,
          inputs=match_data,
          action_type='match',
          api_name=element_selector_api_name,
          xpath=element_selector_xpath,
          currently_executing_code=statement,
          comment='no match found',
          screenshot=self.state.screenshot)
      raise ActionError(f'Fail to match({match_data}) in {self.api_name}[{self.element_list_xpath}]', self.api_name, self.element_list_xpath, 'match', match_data)
      # return None
    else:
      _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=match_data,
        action_type='match',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        comment='action match',
        screenshot=None)
      return found_match # todo:: 

  def __len__(self):
    # get the currently executing code
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, '__len__', self.api_name)
    
    element_selector_api_name = self.api_name if self.api_name else self.element_list_xpath
    element_selector_xpath = self.element_list_xpath
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    target_ele, _ = self.verifier.get_and_navigate_target_element(element_selector_api_name, element_selector_xpath, statement)

    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=None,
        action_type='__len__',
        api_name=element_selector_api_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None)
    
    if target_ele == None: # todo:: maybe it's 0
      logging.warning(f'not found {self.api_name}[{self.element_list_xpath}]')
      return 0
    ele_list_children = target_ele.children
    self.check_action_count()
    return len(ele_list_children)

  def get_current_code_line(self, lineno: int, action: str, element_selector_name: str):
    # get the currently executing code
    code_lines = self.config.compiled_code_lines
    print(
        f"{action}: {element_selector_name if element_selector_name != None else (self.api_name if self.api_name else self.element_list_xpath)} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}"
    )
    current_code_line = code_lines[lineno - 1]
    lineno_in_original_script = self.config.line_mappings[lineno - 1]
    original_code_line = self.config.code_lines[lineno_in_original_script]

    return current_code_line, lineno_in_original_script, original_code_line

  def find_target_element_in_group(self, element_selector_api_name: str, element_selector_xpath: str, statement: dict):
    target_ele_group, _ = self.verifier.get_and_navigate_target_element(self.api_name, self.element_list_xpath, statement)
    target_ele = None
    element_tree = self.element_tree
    subtree = element_tree.extract_subtree(target_ele_group.id)
    if subtree:
      target_ele = subtree.get_ele_by_xpath(element_selector_xpath)

    if target_ele == None:
      _save2log(
          save_path=self.save_path,
          log_file=self.config.log_file,
          element_tree=element_tree,
          idx=target_ele_group.id,
          inputs=None,
          action_type='find_target_element_in_group',
          api_name=element_selector_api_name,
          xpath=element_selector_xpath,
          currently_executing_code=statement,
          comment='crashed',
          screenshot=element_tree.screenshot)
      if self.doc.check_api_name_in_current_screen(element_selector_api_name, self.element_tree.skeleton):
        raise XPathError(f'Not Exist {element_selector_api_name}[{element_selector_xpath}]', element_selector_api_name, element_selector_xpath)
      else:
        raise NotFoundError(f'Not Found {element_selector_api_name}[{element_selector_xpath}] in {self.api_name}[{self.element_list_xpath}]', element_selector_api_name, element_selector_xpath, self.api_name, self.element_list_xpath)
    
    return target_ele
  
  def check_api(self, api, action_type, statement, text=None):
    if isinstance(api, str):
      api_name = api.split('$')[-1]
      try:
        xpath = self.api_xpaths[api_name]
      except KeyError:
        _save2log( # save crash in get_and_navigate_target_element
            save_path=self.save_path,
            log_file=self.config.log_file,
            element_tree=self.element_tree,
            idx=None,
            inputs=text,
            action_type=action_type,
            api_name=api_name,
            xpath=None,
            currently_executing_code=statement,
            comment='crashed',
            screenshot=self.state.screenshot)
        raise APIError(f'Invalid {api_name}', api_name)
    else:
      api_name = api.api_name if api.api_name else api.element_list_xpath
      xpath = api.element_list_xpath
    return api_name, xpath
    
  def _execute_action(self, api_name, xpath, statement, action_type, text: str=None):
    # it is different from the verifier's _execute_action
    target_ele = self.find_target_element_in_group(api_name, xpath, statement)
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=text,
        action_type=action_type,
        api_name=api_name,
        xpath=xpath,
        currently_executing_code=statement,
        comment='action',
        screenshot=self.state.screenshot)
    
    executable_action = agent_utils.convert_action(action_type, target_ele, text)
    self.env.execute_action(executable_action)
    time.sleep(WAIT_AFTER_ACTION_SECONDS)
    self.update_state()
    self.check_action_count()
    
  def tap(self, button_api=None):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'touch', button_api)
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    action_type = 'touch'
    if button_api == None:
      self.verifier._execute_action(self.api_name, self.element_list_xpath, statement, action_type)
      return
    api_name, xpath = self.check_api(button_api, action_type, statement)
    self._execute_action(api_name, xpath, statement, action_type)

  def long_tap(self, button_api):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'long_touch', button_api)
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    action_type = 'long_touch'
    if button_api == None:
      self.verifier._execute_action(self.api_name, self.element_list_xpath, statement, action_type)
      return
    api_name, xpath = self.check_api(button_api, action_type, statement)
    self._execute_action(api_name, xpath, statement, action_type)

  def set_text(self, text, input_api=None):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'set_text', input_api)
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    action_type = 'set_text'
    if input_api == None:
      self.verifier._execute_action(self.api_name, self.element_list_xpath, statement, action_type, text)
      return
    api_name, xpath = self.check_api(input_api, action_type, statement, text)
    self._execute_action(api_name, xpath, statement, action_type, text)

  def get_text(self, element_selector=None):
    '''
    return the text of the element as a string.
    '''
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'get_text', element_selector)
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    action_type = 'get_text'
    if element_selector == None:
      target_ele, _ = self.verifier.get_and_navigate_target_element(self.api_name, self.element_list_xpath, statement)
      element_selector_name = self.api_name 
      element_selector_xpath = self.element_list_xpath
    else:
      if isinstance(element_selector, str):
        element_selector_name = element_selector.split('$')[-1]
        try:
          element_selector_xpath = self.api_xpaths[element_selector_name]
        except KeyError:
          _save2log( # save crash in get_and_navigate_target_element
              save_path=self.save_path,
              log_file=self.config.log_file,
              element_tree=self.element_tree,
              idx=None,
              inputs=None,
              action_type=action_type,
              api_name=element_selector_name,
              xpath=None,
              currently_executing_code=statement,
              comment='crashed',
              screenshot=self.state.screenshot)
          # raise APIError(f'Invalid {element_selector_name}', element_selector_name)
          return None
      else:
        element_selector_name = element_selector.api_name if element_selector.api_name else element_selector.element_list_xpath
        element_selector_xpath = element_selector.element_list_xpath
      
      target_ele = self.find_target_element_in_group(element_selector_name, element_selector_xpath, 'get_text', statement)
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=None,
        action_type=action_type,
        api_name=element_selector_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None)
    self.check_action_count()
    # not change screen
    
    text = target_ele.text if target_ele.text else ''
    text = text.replace('--', ' ')
    return text

  def get_attributes(self, element_selector=None):
    '''
    return the attributes of the element as a dict, dict keys include "selected", "checked", "scrollable", dict values are boolean. eg. get_attributes($files[3])["selected"].
    '''
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    lineno = caller_frame.f_lineno
    current_code_line, lineno_in_original_script, original_code_line = self.get_current_code_line(lineno, 'get_attributes', element_selector)
    statement = {
            'current_code': current_code_line,
            'original_lineno': lineno_in_original_script,
            'original_code': original_code_line
        }
    
    action_type = 'get_attributes'
    if element_selector == None:
      target_ele, _ = self.verifier.get_and_navigate_target_element(self.api_name, self.element_list_xpath, statement)
      element_selector_name = self.api_name 
      element_selector_xpath = self.element_list_xpath
    else:
      if isinstance(element_selector, str):
        element_selector_name = element_selector.split('$')[-1]
        try:
          element_selector_xpath = self.api_xpaths[element_selector_name]
        except KeyError:
          _save2log( # save crash in get_and_navigate_target_element
              save_path=self.save_path,
              log_file=self.config.log_file,
              element_tree=self.element_tree,
              idx=None,
              inputs=None,
              action_type=action_type,
              api_name=element_selector_name,
              xpath=None,
              currently_executing_code=statement,
              comment='crashed',
              screenshot=self.state.screenshot)
          raise APIError(f'Invalid {element_selector_name}', element_selector_name)
      else:
        element_selector_name = element_selector.api_name if element_selector.api_name else element_selector.element_list_xpath
        element_selector_xpath = element_selector.element_list_xpath
      
      target_ele = self.find_target_element_in_group(element_selector_name, element_selector_xpath, 'get_attributes', statement)
    
    _save2log(
        save_path=self.save_path,
        log_file=self.config.log_file,
        element_tree=self.element_tree,
        idx=target_ele.id,
        inputs=None,
        action_type=action_type,
        api_name=element_selector_name,
        xpath=element_selector_xpath,
        currently_executing_code=statement,
        screenshot=None)
    self.check_action_count()
    # not change screen
    target_ele_attrs = target_ele.get_attributes()
    target_ele_attrs['text'] = target_ele_attrs['text'].replace('--', ' ') if target_ele_attrs['text'] else ''
    return target_ele_attrs
  
  def scroll(self, direction, element_selector=None):
    if not element_selector:
      api_name = self.api_name if self.api_name else self.element_list_xpath
      xpath = self.element_list_xpath
      # get the currently executing code
      code_lines = self.config.compiled_code_lines
      frame = inspect.currentframe()
      caller_frame = frame.f_back
      lineno = caller_frame.f_lineno
      print(f"scroll {direction}: {api_name} at line {lineno}, code is:{code_lines[lineno - 1]}, action count: {self.action_count}")
      current_code_line = code_lines[lineno - 1]
      lineno_in_original_script = self.config.line_mappings[lineno - 1]
      original_code_line = self.config.code_lines[lineno_in_original_script]

      last_screen = self.verifier.last_screen
      
      statement = {
          'current_code': current_code_line,
          'original_lineno': lineno_in_original_script,
          'original_code': original_code_line
      }
      if 'up' in direction.lower():
        direction_str = 'up'
      elif 'down' in direction.lower():
        direction_str = 'down'
      elif 'left' in direction.lower():
        direction_str = 'left'
      elif 'right' in direction.lower():
        direction_str = 'right'
      elif 'page_right' in direction.lower():
        direction_str = 'page_right'
      elif 'page_left' in direction.lower():
        direction_str = 'page_left'
      else:
        direction_str = 'down'
      action_type = f'scroll {direction_str}'
      
      self.verifier._execute_action(api_name, xpath, statement, action_type)
      is_to_bottom = False if not last_screen else self.verifier.last_screen == last_screen
      return is_to_bottom

    return self.verifier.scroll(element_selector, direction)

  def back(self):
    self.verifier.back()

  def enter(self):
    self.verifier.enter()