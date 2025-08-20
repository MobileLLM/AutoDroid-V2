"""Agent for executing code script."""

import time
import os
import re
import traceback
import tools as tools
import agent.environment as environment
from agent.script_utils.ui_apis import CodeConfig, CodeStatus, Verifier, regenerate_script, _save2log
from agent.script_utils.bug_processor import BugProcessorV3
from agent.script_utils.solution_generator import SolutionGenerator
from agent.script_utils.api_doc import ApiDoc
from agent.script_utils.err import XPathError


def process_error_info(original_script, compiled_script, traceback, error, error_type,
                       line_mappings, state_xml):
  print(f"Exception caught: {error}")
  print("Traceback details:")
  print(traceback)

  # extract the line number of the error
  tb_lines = traceback.split('\n')

  for line in tb_lines:
    if 'File "<string>"' in line:
      print(line.strip())
      match = re.search(r'line (\d+)', line)
      if match:  # localized the error line number
        try:
          line_number = match.group(1)
          print(f'The line number is: {line_number}')
          error_line_number_in_compiled_script = int(line_number.strip()) - 1  # the line number of the error info often starts from 1, but in the mappings, it starts from 0
          error_line_number_in_original_script = line_mappings[
              error_line_number_in_compiled_script]
          error_line_in_compiled_script = compiled_script.split(
              '\n')[error_line_number_in_compiled_script]
          error_line_in_original_script = original_script.split(
              '\n')[error_line_number_in_original_script]
        except Exception as e:
          print(f'Error in extracting the line number: {line_number}')
          print(e)
          error_line_number_in_compiled_script = None
          error_line_number_in_original_script = None
          error_line_in_compiled_script = None
          error_line_in_original_script = None
      else:
        print('No line number found')
        error_line_in_compiled_script = line
        error_line_in_original_script, error_line_number_in_compiled_script, error_line_number_in_original_script = None, None, None

  return {
      'original_script':
          original_script,
      'compiled_script':
          compiled_script,
      'traceback':
          traceback,
      'error':
          error,
      'error_type':
          error_type,
      'error_line_number_in_compiled_script':
          error_line_number_in_compiled_script,
      'error_line_number_in_original_script':
          error_line_number_in_original_script,
      'error_line_in_compiled_script':
          error_line_in_compiled_script,
      'error_line_in_original_script':
          error_line_in_original_script,
      "error_screen": state_xml
  }


class CodeAgent():
  """code agent"""

  # Wait a few seconds for the screen to stabilize after executing an action.
  WAIT_AFTER_ACTION_SECONDS = 2.0
  MAX_RETRY_TIMES = 2

  FREEZED_CODE = False
  
  def __init__(self,
               env: environment.AsyncEnv,
               app_name: str,
               doc_path: str,
               save_path: str,
               name: str = 'CodeAgent',
               model = None):

    self.env = env
    self.name = name
    app_name = app_name.strip()
    self.model = model
    
    if not os.path.exists(doc_path):
      raise ValueError(f'Unknown doc path: {doc_path}')

    self.app_name = app_name
    self.doc = ApiDoc(doc_path)
    self.save_dir = save_path
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    
    self.save_path = None
    self.code_config = CodeConfig(app_name, self.doc)
    self.code_status = CodeStatus()
    self.current_plan = None
  
  # def __del__(self):
  #   self.doc.save()

  def step(self, goal: str) -> dict:
    """
    only execute once for code script
    """
    task_goal = goal
    app_name = self.app_name
    self.save_path = self.save_dir
    
    tools.dump_json_file(os.path.join(self.save_dir, 'app.json'), 
                         {
                              'app_name': app_name,
                              'doc_path': self.doc.doc_path
                         })
    
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    
    print(f'Executing task: {task_goal}')
    task_info = {
      'goal': task_goal,
      # 'params': str(self.task_params),
      # 'name': self.task_name,
    }
    runtime = []
    
    app_doc = self.doc
    err = None
    done = False
    for retry_time in range(self.MAX_RETRY_TIMES + 1):
      t0 = time.time()
      if self.FREEZED_CODE:
        code = self.code_config.code
      elif retry_time == 0: # first time
        # generate code
        solution_generator = SolutionGenerator(app_name, task_goal, app_doc, self.model)
        solution_code, solution_plan = solution_generator.get_solution(
            prompt_answer_path=os.path.join(self.save_path, f'solution.json'),
            env=self.env,
            model_name=self.model)
        code = solution_code
        self.current_plan = solution_plan
        print(f'Generated code: \n{code}')
      else: # retry
        bug_processor = BugProcessorV3(
            app_name=app_name,
            task=task_goal,
            doc=app_doc,
            error_info=error_info,
            code=code,
            previous_plan = self.current_plan,
            err = err)
        
        # update the save_path for retry
        self.save_path = os.path.join(self.save_dir, f'{retry_time}')
        os.makedirs(self.save_path, exist_ok=True)
        
        if isinstance(err, XPathError):
          bug_processor.fix_invalid_xpath(
            env=env, 
            api_name=err.name, 
            prompt_answer_path=os.path.join(self.save_path, f'fix_xpath.json'), 
            model_name=model)
        
        code, plan = bug_processor.get_fixed_solution(# re-generate code
            prompt_answer_path=os.path.join(self.save_path, f'solution.json'),
            env=env,
            model_name=model)
        self.current_plan = plan
        self.env.reset_env(self.app_name, True)
        print(f'Generated code: \n{code}')

      tools.dump_json_file(f'{self.save_path}/task.json', task_info)
      if isinstance(code, list):
        code = '\n'.join(code)
      tools.write_txt_file(f'{self.save_path}/code.txt', code)
      
      code_script, line_mappings = regenerate_script(code, 'verifier')
      tools.write_txt_file(f'{self.save_path}/compiled_code.txt', code_script)
      tools.dump_json_file(f'{self.save_path}/line_mappings.json', line_mappings)
      
      # in case some silly scripts include no UI actions at all, we make an empty log for batch_verifying
      tools.dump_yaml_file(os.path.join(self.save_path, f'log.yaml'), {'records': [], 'step_num': 0})
      
      env = self.env
      self.code_config.set(self.save_path, code, code_script, line_mappings)
      self.code_status.reset()
      
      t1 = time.time()
      # execution
      verifier = Verifier(env, self.code_config, self.code_status)
      
      try:
        exec(code_script)
        done = True
        t2 = time.time()
        runtime.append({
            'total': t2 - t0,
            'solution': t1 - t0,
            'execution': t2 - t1
        })
        break
      except Exception as e:
        tb_str = traceback.format_exc()
        ex_type = type(e).__name__
        error_info = process_error_info(code, code_script, tb_str, str(e), ex_type,
                                        line_mappings, self.env.get_state().element_tree.str)

        error_path = os.path.join(self.save_path, f'error.json')
        tools.dump_json_file(error_path, error_info)
        err = e
    
    result = {
      'is_completed': done,
      'save_path': self.save_path,
      'runtime': runtime,
      "task": goal,
      "app": app_name,
      "code": code
    }
    
    if result['is_completed']:
        self.env.execute_action({
            'action_type': 'STATUS_TASK_COMPLETE',
        })
    else:
        self.env.execute_action({
            'action_type': 'STATUS_TASK_IMPOSSIBLE',
        })    
        print(f'Failed task: \n{task_goal}')

    _save2log(
        save_path=self.save_path,
        log_file=self.code_config.log_file,
        element_tree=verifier.element_tree,
        idx=-1,
        inputs=None,
        action_type=None,
        api_name=None,
        xpath=None,
        currently_executing_code=None,
        comment='done',
        screenshot=None)
    
    tools.dump_json_file(f'{self.save_path}/runtime.json', runtime)
    tools.dump_json_file(f'{self.save_path}/agent_actions_save_time.json', self.env.actions_taken)
    tools.dump_json_file(f'{self.save_path}/gpt_located_elements.json', verifier.gpt_located_elements)
    tools.dump_json_file(f'{self.save_dir}/result.json', result)

    return result
