from copy import deepcopy
import pandas
import os
import re
import sys
import json
import time
import logging
import traceback
import pkg_resources
import shutil
import subprocess
import agent.environment as environment
import tools as tools
from evaluation.droidtask.config import BASE_APK_PATH, DEBUG_MODE, DOC_PATH, EMULATOR_AGRS, FIRST_SCREEN_ELEMENTS_PATH, TASKS_PATH
from evaluation.droidtask.experiment.query_llm import make_solution_prompt_droidtask_tune
from agent.droidbot.device import Device
from agent.droidbot.app import App
from agent.droidbot.input_event import RestartAppEvent
from agent.code_agent import CodeAgent
from agent.script_utils.ui_apis import CodeConfig, CodeStatus, Verifier, regenerate_script, _save2log, ElementList
from agent.script_utils.api_doc import ApiDoc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def process_error_info(original_script, compiled_script, traceback, error,
                       line_mappings):
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
          error_line_number_in_compiled_script = int(
              line_number
          ) - 1  # the line number of the error info often starts from 1, but in the mappings, it starts from 0
          error_line_number_in_original_script = line_mappings[
              error_line_number_in_compiled_script]
          error_line_in_compiled_script = compiled_script.split(
              '\n')[error_line_number_in_compiled_script]
          error_line_in_original_script = original_script.split(
              '\n')[error_line_number_in_original_script]
        except:
          print(f'Error in extracting the line number: {line_number}')
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
      'error_line_number_in_compiled_script':
          error_line_number_in_compiled_script,
      'error_line_number_in_original_script':
          error_line_number_in_original_script,
      'error_line_in_compiled_script':
          error_line_in_compiled_script,
      'error_line_in_original_script':
          error_line_in_original_script
  }
  

class CodeAgent():
  """code agent"""

  # Wait a few seconds for the screen to stabilize after executing an action.
  WAIT_AFTER_ACTION_SECONDS = 2.0
  MAX_RETRY_TIMES = 2
  
  def __init__(self,
               env: environment.AsyncEnv,
               app_name: str,
               doc_path: str,
               save_path: str,
               name: str = 'CodeAgent'):

    self.env = env
    self.name = name
    app_name = app_name.strip()
    
    if not os.path.exists(doc_path):
      raise ValueError(f'Unknown doc path: {doc_path}')

    self.app_name = app_name
    self.doc = ApiDoc(doc_path)
    self.save_dir = save_path
    
    self.save_path = None
    self.code_config = CodeConfig(app_name, self.doc)
    self.code_status = CodeStatus()

  def run_code(self, code: str) -> dict:
    """
    only execute once for code script
    """
    app_name = self.app_name
    self.save_path = self.save_dir
    
    tools.dump_json_file(os.path.join(self.save_dir, 'app.json'), 
                         {
                              'app_name': app_name,
                              'doc_path': self.doc.doc_path
                         })
    
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    self.env.execute_action(
        {
            'action_type': 'open_app',
            'app_name': app_name
        })
    time.sleep(self.WAIT_AFTER_ACTION_SECONDS)
    runtime = []
    
    done = False
    t0 = time.time()
    
    tools.dump_json_file(f'{self.save_path}/task.json', {})
    if isinstance(code, list):
      code = '\n'.join(code)
    tools.write_txt_file(f'{self.save_path}/code.txt', code)
    # import pdb;pdb.set_trace()
    code_script, line_mappings = regenerate_script(code, 'verifier')
    print(code_script)
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
      error_info = None
    except Exception as e:
      done = False
      tb_str = traceback.format_exc()
      error_info = process_error_info(code, code_script, tb_str, str(e),
                                      line_mappings)

      error_path = os.path.join(self.save_path, f'error.json')
      tools.dump_json_file(error_path, error_info)
    
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
    t2 = time.time()
    runtime.append({
        'total': t2 - t0,
        'solution': t1 - t0,
        'execution': t2 - t1
    })

  
    result = {
        'is_completed': done,
        'save_path': self.save_path,
        'runtime': runtime,
        "error_info": error_info
      }
    
    tools.dump_json_file(f'{self.save_dir}/result.json', result)

    return result


def check_code_executable(app_name, code, task_id, output_dir):
    output_dir = f"{output_dir}/{app_name}/{task_id}"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    # app_name = "contacts"
    doc_name = f"{DOC_PATH}/{app_name}.json"
    app_path = f"{BASE_APK_PATH}/{app_name}.apk"

    if output_dir is not None:
      if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
      html_index_path = pkg_resources.resource_filename("droidbot", "resources/index.html")
      stylesheets_path = pkg_resources.resource_filename("droidbot", "resources/stylesheets")
      target_stylesheets_dir = os.path.join(output_dir, "stylesheets")
      if os.path.exists(target_stylesheets_dir):
        shutil.rmtree(target_stylesheets_dir)
      shutil.copy(html_index_path, output_dir)
      shutil.copytree(stylesheets_path, target_stylesheets_dir)

    logging.info("Starting DroidBot")
    try:
      device_serial = f"emulator-{EMULATOR_AGRS['port']}"
      subprocess.run(["adb",  "-s", device_serial, "emu", "avd", "snapshot", "load", EMULATOR_AGRS['snapshot']])  
      import time
      time.sleep(3)
      
      device = Device(
          device_serial=device_serial,
          is_emulator=True,
          output_dir=output_dir)
      
      app = App(app_path, output_dir)
      env = environment.AsyncDroidBotEnv(device, app)
      device.send_event(RestartAppEvent(app=app))
      if 'firefox' in app_name.lower():
        import time
        time.sleep(30)
      code_agent = CodeAgent(env, app_name, doc_name, output_dir)
      
      print(task_id)
      # code_agent.FREEZED_CODE = True
      code_agent.MAX_RETRY_TIMES = 2
      code_agent.code_config.enable_dependency = False # now document don't support dependency format
      
      # run task
      result = code_agent.run_code(code)
      
      print(result)

    except KeyboardInterrupt:
      logging.info("Keyboard interrupt.")
      device.disconnect()
      sys.exit(0)
    except Exception:
      import traceback
      traceback.print_exc()
      device.disconnect()
      return {'failed': True}
      # sys.exit(-1)

    logging.info("DroidBot Stopped")
    device.disconnect()

    return result

def remove_scroll_lines(code):
  code_lines = code.split("\n")
  removed_code_lines = []
  for line in code_lines:
    if ".scroll(" not in line:
      removed_code_lines.append(line)
    elif line.lstrip().startswith("#"):
      removed_code_lines.append(line)
  new_code = "\n".join(removed_code_lines)
  return new_code
  
def remove_quotes(code):
  if code.lstrip().startswith('"""') or code.lstrip().startswith("'''"):
    code = code[3:]
  if code.rstrip().endswith('"""') or code.rstrip().endswith("'''"):
    code = code[:-3]
  return code

def postprocess_code(code, doc):
  code = re.sub(r'([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)', r'$\1__\2', code)
  code = code.replace("$$", "$")# code = "$main_screen__new_note_button.tap()\n$add_new_note_dialog__cancel_button.tap()"
  code = remove_scroll_lines(code)
  code = remove_quotes(code)
  return code

def load_autodroidv2():
    model_path = "autodroidv2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer
    
def query_autodroidv2(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_all_tasks(app, output_dir, model_name="autodroidv2"):
  
  tasks_data = tools.load_json_file(TASKS_PATH)
  
  result_folder = 'results'
  
  if model_name == "autodroidv2":
    model, tokenizer = load_autodroidv2()

  import tiktoken
  encoder = tiktoken.get_encoding("cl100k_base")
  
  for app_name, app_tasks in tasks_data.items():
    if app_name not in [app]:
      continue
    print("app_name:", app_name)

    result_list = []
    for task_number, task_data in app_tasks.items():
      try:
        task = task_data['task']
        
        first_screen_elements = tools.load_json_file(f'{FIRST_SCREEN_ELEMENTS_PATH}/{app_name}_first_elements.json')
        doc = tools.load_json_file(f'{DOC_PATH}/{app_name}.json')
        if DEBUG_MODE:
              code = '''
# $server_overview_screen__you_button.tap()
# $personal_profile_screen__settings_button.tap()
# $settings_screen__notifications_button.tap()
# '''
        else:
          task_prompt = make_solution_prompt_droidtask_tune(doc, task, app_name, first_screen_elements)
          print(task_prompt)
          if model_name == "autodroidv2":
            task_answer = query_autodroidv2(model, tokenizer, task_prompt)
          else:
            task_answer = tools.query_model(task_prompt, model_name)
          print(task_answer)
          if not os.path.exists(f'{output_dir}/{app_name}'):
            os.makedirs(f'{output_dir}/{app_name}')
          tools.dump_json_file(json_path=f'{output_dir}/{app_name}/{task_number}_qa.json', data=[task_prompt, task_answer])
          # calculate the token number of the answer, if too long, we skip
          tokens = encoder.encode(task_answer)
          if len(tokens) > 2048:
            print(f"Task answer too long: {len(tokens)}")
            continue
          
          task_answer, _ = tools.convert_gpt_answer_to_json(task_answer, 'gpt-4o')
          code = task_answer['script']
        
        
        code = postprocess_code(code, doc)
        print(f"Post processed code: {code}")
        result = check_code_executable(
          app_name=app_name,
          code=code,
          task_id=task_number, 
          output_dir=output_dir
        )
        result.update({"task": task, 'code': code, 'doc_path': f'{DOC_PATH}/{app_name}.json', })
      except KeyboardInterrupt as e:
        print(e)
        sys.exit(-1)
      except Exception as e:
        traceback.print_exc()
        print(e)
        result = {'failed': True}

      result_list.append(deepcopy(result))
      result_df = pandas.DataFrame(result_list)
      if not os.path.exists(f'{output_dir}/{result_folder}'):
        os.makedirs(f'{output_dir}/{result_folder}')
      output_path = os.path.join(output_dir, result_folder, f"{app_name}.jsonl")
      result_df.to_json(output_path, lines=True, orient='records')
      if DEBUG_MODE:
        sys.exit(1)