import tools as tools
import agent.environment as environment
from agent.script_utils.api_doc import ApiDoc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


ACTIONS_DSL_PROMPT_DESCRIPTION = '''
You should follow the guidelines below to complete the task:
In the script, except for the common python control flow (for, if-else, function def/calls, etc.), you can use the following APIs:
- Use '$' to reference an element API, example: '$calendar_screen__date_today'.
- <element_selector>.tap() -> None: tap on the element. Almost all elements can be taped. If an element's attribute checked=false or selected=false, tapping it can make it checked or selected, vice versa. Usage: e.g. $device_theme_settings__dark_theme_button.tap()
- <element_selector>.tap(<child_element>) -> None: tap on the child element of an element. Usage: e.g. $search_screen__search_results_list_item.tap($search_screen_search_result_item_options_button)
- <element_selector>.long_tap() -> None: long tap the element. Usage: e.g. $calendar_screen__date_today.long_tap()
- <element_selector>.set_text(<text>) -> None: set the text of the element to <text>. Only editable text fields can be set text. Usage: e.g. $home_screen__search_input_field.set_text("volleyball photos")
- <element_selector>.scroll(<direction>) -> bool: scroll the UI element in the specified direction, and return whether scroll to the bottom in the screen, and direction is a str from "up", 'down", "left", "right". You can also use "page_up", "page_down", "page_left", "page_right" as scrolling direction because some UI screens do not have scrollable elements available but allow the whole page to be scrolled. Usage: e.g. $scroll_settings_page.scroll("down")
- <element_selector>.get_text() -> str | None: return the text of the element as a string or None if the element does not have any text. Usage: e.g. $home_screen__search_input_field.get_text()
- <element_selector>.get_attributes() -> dict[str, str]: return the attributes of the element as a dict, dict keys include "selected", "checked", "scrollable", dict values are boolean. eg. $files[3].get_attributes()["selected"].
- back() -> None: use back to return to the previous screen. It can be used when there is no obvious element for navigating backwards in order to return to previous screen. Usage: back()
- enter() -> None: use enter when you want to submit a search query after you filled the search input and you want to trigger the searching action. Usage: enter()


The <element_selector> primitive is used to select an element, possible ways of selection include:
- <element id>, eg. $settings_button
- <element_list>[<idx>] -> UI element: the idx-th in the element list. eg. $my_items[1]

The <element_list> primitive is used to select a list of elements, possible ways of selection include:
- <element_selector>: the items in the list element identified by <element_selector>. eg. $my_items
- <element_list>.match(<text or attribute dict>) -> UI element: the element in the element list that matches the given text or attribute dict. eg. $my_items.match("key words") or $my_items.match({{'selected': True}}).
You can use len($<element_list>) -> int: to get the total number of items in an element list.

- The variable assignment must follow the format: <variable_name> = $<element_selector>.<action> or $<element_list>. Usage: e.g. my_variable = $home_screen__search_input_field.get_text()

- Examples: 

```python
# Task: Search for volleyball photos in the pictures section
$home_screen__pictures_button.tap()
$pictures_screen__search_bar.tap()
$search_screen__search_input_field.set_text("volleyball photos")
```

```python
# Task: Delete all events in Calendar
$calendar_screen__more_options_button.tap()
$more_options_popup__options_list.match("Settings").tap()
$settings_interface__delete_all_events_button.tap()
$confirmation_dialog_delete_all_events__yes_button.tap()
```

Above are examples of how to use the provided APIs. Please focus on the actual user task!
'''

class SolutionGenerator:

  def __init__(self, app_name: str, task: str, doc: ApiDoc, model_name:str):
    self.app_name = app_name
    self.task = task
    self.doc = doc
    self.model_name = model_name
    if model_name == "autodroidv2":
      self.load_autodroidv2()
    
  def make_prompt(self, env: environment.AsyncEnv):
    # all elements
    all_elements_desc = self.doc.get_all_element_desc()
    
    # current screen
    state = env.get_state()
    # current screen elements
    current_screen_desc = self.doc.get_current_element_desc(state)

    element_tree = state.element_tree
    visible_html_view = element_tree.get_str_with_visible()
    
    return f'''Imagine that you are a robot operating a smartphone to use the {self.app_name} app. Like how humans operate the smartphone, you can tap, long tap, input text, scroll, and get attributes of the UI elements in the {self.app_name} app. However, unlike humans, you cannot see the screen or interact with the physical buttons on the smartphone. Therefore, you need to write scripts to manipulate the UI elements (buttons, text fields, scrollers, element_lists, etc) in the app. 

You are provided with: 
1. Your ultimate task to be completed using the app. 
2. A description of the UI elements in the app, which includes: 
    - The name of an element is in the format of: <screen_name>__<element_name>, such as main_screen__recipe_title. The screen_name is a concise description of the screen where the element is located, and the element_name distinguish those elements in same screen.
    - Options: Some of the element lists have options as reference, you can match the element with the options by <element_selector>.match(<option>)
    - Effect: the effect of interacting with the element, including which UI screen will be shown or what will be changed, which elements will be accessed, etc.


You should output a python-style code to complete the task.
- The script must be an executable python code with valid intents and logic. 
- Do not wrap the code in a function, it must be executed right away. 
    



And here is the start screen of the app described by HTML:
{visible_html_view}

Now, {ACTIONS_DSL_PROMPT_DESCRIPTION}

**Your ultimate task is: {self.task}**

You can use the following important UI elements:
{all_elements_desc}


Your answer should follow this JSON format:

{{
    "plan": "<string, revised high level plan to complete the task; addressing the problems in the previous plan>",
    "elements": "<string, analysis on the elements that could be used to complete the task; why each element is suitable for this task>", 
    "script": "<string, the python script to complete the task>",
}}

Note that: 
- **you should only output the JSON content.**
- **you must use '$' only before any UI element**
- **pay attention to save the changes to the app settings, if save appears in the UI**'''
  
  def load_autodroidv2(self):
      model_path = "autodroidv2"
      tokenizer = AutoTokenizer.from_pretrained(model_path)
      model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
      return model, tokenizer
      
  def query_autodroidv2(self, prompt: str):
      inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
      output = self.model.generate(**inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
      return self.tokenizer.decode(output[0], skip_special_tokens=True)

  def get_solution(self,
                   prompt_answer_path: str,
                   env: environment.AsyncEnv,
                   model_name='gpt-4o'):
    prompt = self.make_prompt(env)
    print("Query GPT-4o for solution")
    # write the prompt to a txt file concately
    tools.append_to_txt_file(prompt_answer_path.replace('.json', '.txt'), f'{prompt}\n'+('='*50)+'\n\n')
    if self.model_name == "autodroidv2":
      answer = self.query_autodroidv2(prompt)
    else:
      answer = tools.query_model(model=model_name, prompt=prompt)

    tools.append_to_txt_file(prompt_answer_path.replace('.json', '.txt'), f'{answer}\n'+('*'*50)+'\n\n')
    answer, tokens = tools.convert_gpt_answer_to_json(
        answer, model_name=model_name, default_value={
            'Plan': '',
            'Script': ''
        })
    tools.dump_json_file(prompt_answer_path, {
        'prompt': prompt,
        'answer': answer,
        'convert_tokens': tokens
    })
    if 'Script' in answer.keys():
      return answer['Script'], answer['Plan']
    else:
      return answer['script'], answer['plan']
