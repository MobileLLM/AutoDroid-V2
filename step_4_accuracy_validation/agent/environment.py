import abc
import json
import logging
import os
import time
import dataclasses

import traceback
from typing import Any, Iterator

import pandas as pd

from agent.droidbot.device_state import ElementTree
from agent.emulator_controller import EmulatorController

@dataclasses.dataclass(frozen=True)
class State():
  """State of the Android environment.

  Attributes:
    pixels: RGB array of current screen.
    forest: Raw UI forest; see android_world_controller.py for more info.
    ui_elements: Processed children and stateful UI elements extracted from
      forest.
  """

  screenshot: Any
  element_tree: ElementTree

  @classmethod
  def create_and_infer_elements(
      cls,
      screenshot: Any,
      element_tree: ElementTree,
  ):
    """Creates a new instance, inferring UI elements from the forest."""

    return cls(screenshot, element_tree)

class AsyncEnv(abc.ABC):
  """Interface for interacting with a real-time Android device.

  Computing environments, such as Android, run in real-time, independently of
  the agent interacting with it. All observations and actions are asynchronous
  and OS does not pause when providing observations or when accepting actions.
  Changes from action execution may take some time to appear.
  """

  @abc.abstractmethod
  def reset(self, go_home: bool = False) -> State:
    """Go home on reset.

    Args:
      go_home: Whether to go home during the reset.
    """

  @abc.abstractmethod
  def reset_env(self, wipe_intermidiate_task_data: bool = False) -> State:
    """Reconnect device and reset env variables.

    Args:
      wipe_intermidiate_task_data: Whether to delete the intermidiate files (screenshot, activity, action, view - logs) created during the script execution steps.
    """

  @abc.abstractmethod
  def get_state(self) -> State:
    """Gets the state of the environment; i.e., screenshot path & UI tree.

    In practice this will usually be called after executing an action. Logic
    should be implemented, perhaps a simple time.sleep, to ensure the
    environment updates after the action.

    Returns:
      Observation containing RGB array of screen, the accessibility forest,
        and UI elements derived from the forest. See android_world_controller.py
        for
        more detail.
    """
  def wait_for_stable_state(
        self,
        stability_threshold: int = 3,
        check_interval: float = 0.5,
        timeout: float = 6.0,
    ) -> State:
      """Checks if the UI elements remain stable over a number of checks and gets state.

      Args:
          stability_threshold: Number of consecutive checks where UI elements must
            remain the same to consider UI stable.
          check_interval: Time in seconds to wait between checks.
          max_checks: Maximum number of checks to perform for UI to become stable before
            giving up.

      Returns:
          True if UI is considered stable, False if it never stabilizes within the
          timeout.
      """
  @abc.abstractmethod
  def execute_action(self, action: dict) -> None:
    """Executes action on the environment."""

  @property
  @abc.abstractmethod
  def foreground_activity_name(self) -> str:
    """Returns the activity name of the app currently opened in foreground."""
  
  @property
  @abc.abstractmethod
  def device_screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment in pixels: (width, height)."""
  
  @property
  @abc.abstractmethod
  def actions_taken(self) -> list:
    """Returns the actions taken in the environment with the time it took to save the result of each action."""

  @property
  @abc.abstractmethod
  def logical_screen_size(self) -> tuple[int, int]:
    """Retrieves the logical screen size of the Android device.

    While the physical size is a fixed attribute of the display, the logical
    size is flexible and varies based on system settings such as the orientation
    or if the resolution is changed.

    Returns: The (width, height) in pixels, denoting the logical dimensions of
    the screen. Width and height values are aligned with the device's current
    orientation, meaning width is always logical horizontal direction (like in
    the landscape orientation width will be the physical vertical direction).
    """
    
  @abc.abstractmethod
  def close(self) -> None:
    """Closes the environment."""
    
from agent.droidbot.device import Device
from agent.droidbot.app import App
from agent.droidbot.device_state import DeviceState
from agent.droidbot import input_event

class AsyncDroidBotEnvForLlamaTouch(AsyncEnv):
  
  def __init__(self, avd_name = None, emulator_controller_args=None,\
                 max_steps=30,
                 local_output_path="experiment",
                 instruction_fp="instructions/llamatouch_task_metadata.tsv",
                 config = None):
    self.config = config
    self.device_serial = f"emulator-{emulator_controller_args['port']}"
    self._prior_state = None
    self.local_output_path = local_output_path
    self.device_logs = f"{local_output_path}/device_logs"
    self.logger = logging.getLogger(self.__class__.__name__)
    
    self.emulator_controller = EmulatorController(avd_name=avd_name,device_serial=self.device_serial,params=emulator_controller_args)

    self._state: DeviceState = None
    self._element_tree: ElementTree = None
    self.state_history = []

    self.instructions = pd.read_csv(instruction_fp, sep='\t')
    self.instruction_generator = self._generate_instruction()

    self.max_steps = max_steps
    self.current_action = "None|None|None"
    self._actions_taken = []
    # self.home_screen = ""
    self.app_name = ""

    self.screenshot_dir_path = ""
    self.activity_dir_path = ""
    self.vh_dir_path = ""
    self.vh_json_dir_path = ""
    self.action_dir_path = ""
    self.ep_installed_dir = ""

    self.set_up()

  @property
  def foreground_activity_name(self) -> str:
    return self.device.get_top_activity_name()
  
  @property
  def device_screen_size(self) -> tuple[int, int]:
    return self._state.width, self._state.height

  @property
  def actions_taken(self) -> list:
    return self._actions_taken
  
  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return self.device_screen_size
  
  def stop_app(self) -> None:
     self.device.stop_app(self.app_name)
     
  def set_up(self) -> None:
    self.logger.info("loading emulator...")
    self.emulator_controller.load_emulator_with_snapshot()
    time.sleep(30) # waiting for emulator to start
    self.logger.info("connecting to device...")
    self.device = Device(
        is_emulator=True,
        device_serial=self.device_serial,
        output_dir=self.device_logs)
    self.device.set_up()
    self.device.connect()
    time.sleep(2)
    self.logger.info("AgentEnv setup over!")

  def reset(self, go_home: bool = False) -> State:
    if go_home:
      self.device.send_event(input_event.KeyEvent('HOME'))
    return self.get_state()
  
  def reset_env(self, app_name, wipe_intermidiate_task_data = False):
    if wipe_intermidiate_task_data and len(self.actions_taken) == 0:
      return
    
    self.logger.info("resetting agent env...")
    self.current_action = "None|None|None"
    self.state_history = []
    self.episode_end = False
    self._actions_taken = []
    self.app_name = app_name
    self.device.disconnect()
    time.sleep(15)
    # if wipe_intermidiate_task_data:
    #   self.logger.info("wiping intermidiate results...")
    #   # os.system(f"del  -rf {self.task_output_path}/*")
    #   os.system(f"rm -rf {self.task_output_path}/*")
    #   self.logger.info("intermidiate results wiped successfully!")

    # if self.app_name != app_name or self.home_screen != self._element_tree.str:
    self.emulator_controller.reload_snapshot(self.config.EMULATOR_CONTROLLER_AGRS["snapshot"])
    time.sleep(15)
    self.device.set_up()
    self.device.connect()
    time.sleep(5)
    self.prepare(app_name)
      
    self.logger.info("agent env reset successfully!")
  
  def prepare(self, app_name = None):
    if app_name != None:
      self.app_name = app_name

    self._setup_directories(self.task_output_path)
    apk_path = f"{self.config.BASE_APKS_PATH}/{self.config.APKS_PER_APP[self.app_name]}"
    app = App(apk_path, f"{self.device_logs}/{self.app_name}")
    
    self.device.send_event(input_event.RestartAppEvent(app=app))
    self.device.start_app(app)
    time.sleep(10)
    self._update_state()

  def close(self) -> None:
    self.logger.info(f"tear down the agent env...")
    self.device.disconnect()
    time.sleep(5)
    self.emulator_controller.exit_emulator()

  def get_instruction(self) -> str:
    try:
        instruction, path, app = next(self.instruction_generator)
        self.task_output_path = path
        return instruction, app
    except Exception:
        self.logger.warning("All instructions have been fetched.")  
        return None, None

  def execute_action(self, action: dict) -> None:
    event = None
    action_type: str= action['action_type']
    action_params = None
    width, height = self.device_screen_size
    time_spent_locating = None
    if 'time_spent_locating' in action:
      time_spent_locating = action['time_spent_locating']
    if action_type == 'click':
      event = input_event.TouchEvent(action['x'], action['y'])
      action_params = f"[{action['x']/width} {action['y']/height}]"
    elif action_type == 'long_press':
      event = input_event.LongTouchEvent(action['x'], action['y'], duration=1000)
      action_params = f"[{action['x']/width} {action['y']/height}]"
    elif action_type == 'input_text':
      event = input_event.SetTextEvent(action['x'], action['y'], text=action['text'])
      action_params = f"{action['text']}"
    elif action_type == 'scroll':
      event = input_event.ScrollEvent(view=action['view'], direction=action['direction'])
      start_x,start_y,end_x,end_y = event.get_scroll_coordinates(self.device)
      action_params = f"[{start_x/width} {start_y/height}]|[{end_x/width} {end_y/height}]"
    elif action_type == 'navigate_home':
      event = input_event.KeyEvent(name='HOME')
    elif action_type == 'enter':
      event = input_event.KeyEvent(name='ENTER')
    elif action_type == 'back':
      event = input_event.KeyEvent(name='BACK')
    if event != None:
      self.device.send_event(event)
    
    t1 = time.time()
    formatted_action = self._trans_action_format(action_type, action_params, width, height)
    try:
      self._dump_action_state(formatted_action)
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Exception caught: {e}")
        print("Traceback details:")
        print(tb_str)
    t2 = time.time()
    self.actions_taken.append({"action":formatted_action, "log_time":t2 - t1, "location_time": time_spent_locating})

  def _trans_action_format(self, action_type, action_para, width, height) -> Any:
      if action_type == "click" or action_type == "long_press":
          return f"CLICK|{str(action_para)}|NULL|{width}|{height}"
      elif action_type == "scroll":
          return f"SWIPE|{str(action_para[:2])}|{str(action_para[2:])}|{width}|{height}"
      elif action_type == "input_text":
          return f"TYPE|{action_para}|NULL|{width}|{height}"
      elif action_type == "back":
          return f"PRESS_BACK|NULL|NULL|{width}|{height}"
      elif action_type == "enter":
          return f"PRESS_ENTER|NULL|NULL|{width}|{height}"
      elif action_type == "navigate_home":
          return f"PRESS_HOME|NULL|NULL|{width}|{height}"
      elif action_type == "PRESS_ENTER":
          return f"{action_type}|NULL|NULL|{width}|{height}"
      elif action_type == "STATUS_TASK_COMPLETE" or action_type == "STATUS_TASK_IMPOSSIBLE":
          return f"{action_type}|NULL|NULL|{width}|{height}"
      else:
          raise ValueError("action_type not supported")
      
  def _dump_action_state(self, formatted_action):
        tag = len(self.actions_taken)
        view_hierarchy_json_path = os.path.join(self.vh_json_dir_path, f"{tag}.vh")
        activity_path = os.path.join(self.activity_dir_path, f"{tag}.activity")
        action_path = os.path.join(self.action_dir_path, f"{tag}.action")
        ep_installed_fp = os.path.join(self.ep_installed_dir, "installed_apps.txt")

        with open(ep_installed_fp, 'w') as file:
          # Intentionally set to '' as we ignore tasks related to installing applications
            file.write("")
            
        self._do_dump_hierarchy(tag, self.vh_dir_path)   

        with open(view_hierarchy_json_path, "w", encoding="utf-8") as vh_json_file:
            json.dump(self._state.views, vh_json_file, ensure_ascii=False, indent=4)
        
        with open(activity_path, "w", encoding="utf-8") as activity_file:
            activity_file.write(self.foreground_activity_name)
        
        with open(action_path, "w", encoding="utf-8") as action_file:
            action_file.write(formatted_action)
        
        self.device.take_screenshot(self.screenshot_dir_path,tag)
  
  def get_state(self) -> State:
    # if self._element_tree is None:
    self._update_state()
      # self.home_screen = self._element_tree.str
    # if wait_to_stabilize:
    #   return self._get_stable_state()
    return State.create_and_infer_elements(screenshot=self._state.screenshot_path, element_tree=self._element_tree)
  
  def _update_state(self) -> State:
    self._state = self.device.get_current_state()
    _, _, element_tree = self._state.text_representation
    self._element_tree = element_tree

  def wait_for_stable_state(
      self,
      stability_threshold: int = 3,
      check_interval: float = 0.5,
      timeout: float = 6.0,
  ) -> State:
    """Checks if the UI elements remain stable over a number of checks and gets state.

    Args:
        stability_threshold: Number of consecutive checks where UI elements must
          remain the same to consider UI stable.
        sleep_duration: Time in seconds to wait between checks.
        timeout: Maximum time in seconds to wait for UI to become stable before
          giving up.

    Returns:
        True if UI is considered stable, False if it never stabilizes within the
        timeout.
    """
    if self._state is None:
      self.get_state()

    stable_checks = 0
    elapsed_time = 0
    prioir_element_tree = self._element_tree

    while stable_checks < stability_threshold and elapsed_time < timeout:
      try:
        self._update_state()
        if prioir_element_tree.str == self._element_tree.str:
          stable_checks += 1
          if stable_checks == stability_threshold:
            print("State updated!")
            break  # Exit early if stability is achieved.
        else:
          stable_checks = 0  # Reset if any change is detected
          prioir_element_tree = self._element_tree

        time.sleep(check_interval)
        elapsed_time += check_interval
        
      except Exception as e:
        time.sleep(check_interval)
        elapsed_time += check_interval
        print("Error getting state! Trying again..",e)

  
  def _do_dump_hierarchy(self, name, dump_location) -> str:
        device_dump_location = f"/sdcard/{name}.xml"
        self.emulator_controller.run_adb_command(f"shell uiautomator dump {device_dump_location}")
        self.emulator_controller.run_adb_command(f"pull {device_dump_location} {dump_location}")
        # read the content from the dumped file 
        with open(f"{dump_location}/{name}.xml", "r", encoding="utf-8") as file:
            content = file.read()
        if content == "":
            raise Exception("dump hierarchy is empty")
        
        # '<?xml version=\'1.0\' encoding=\'UTF-8\' standalone=\'yes\' ?>\r\n<hierarchy rotation="0" />'
        if '<hierarchy rotation="0" />' in content:
            logging.debug("dump empty, call clear_traversed_text and retry")
            # self.clear_traversed_text()
            raise Exception("dump hierarchy is empty with no children")
        return content
  
  def _generate_instruction(self) -> Iterator[tuple[str, str]]:
    for _, row in self.instructions.iterrows():
        yield row['description'], os.path.join(self.local_output_path, str(row['path'])), row['app']
  
  def _setup_directories(self, base_path) -> list[str]:
    paths = []
    for subdir in ['screenshot', 'activity', 'xml', 'vh', 'action', 'installed_apps']:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        paths.append(dir_path)

    screenshot_dir_path, activity_dir_path, vh_dir_path, vh_json_dir_path, action_path, installed_apps_path = paths
    self.screenshot_dir_path = screenshot_dir_path
    self.activity_dir_path = activity_dir_path
    self.vh_dir_path = vh_dir_path
    self.vh_json_dir_path = vh_json_dir_path
    self.action_dir_path = action_path
    self.ep_installed_dir = installed_apps_path
    return paths
  

class AsyncDroidBotEnv(AsyncEnv):
  
  def __init__(self, device: Device, app: App):
    self.device = device
    self.app = app
    self._prior_state = None
    
    self.device.set_up()
    self.device.connect()
    
    self._state: DeviceState = None
  
  @property
  def state(self) -> DeviceState:
    if not self._state:
      self._state = self.device.get_current_state()
    return self._state
    
  def reset(self, go_home: bool = False) -> State:
    if go_home:
      self.device.send_event(input_event.KeyEvent('HOME'))
    return self.get_state()
  
  def get_state(self, wait_to_stabilize: bool = False) -> State:
    if wait_to_stabilize:
      return self._get_stable_state()
    return self._get_state()

  def execute_action(self, action: dict) -> None:
    event = None
    action_type: str= action['action_type']
    if action_type == 'click':
      event = input_event.TouchEvent(action['x'], action['y'])
    elif action_type == 'long_press':
      event = input_event.LongTouchEvent(action['x'], action['y'], duration=1000)
    elif action_type == 'input_text':
      event = input_event.SetTextEvent(action['x'], action['y'], text=action['text'])
    elif action_type == 'scroll':
      event = input_event.ScrollEvent(view=action['view'], direction=action['direction'])
    elif action_type == 'navigate_home':
      event = input_event.KeyEvent(name='HOME')
    elif action_type == 'navigate_back': # 是navigate_back而不是back
      event = input_event.KeyEvent(name='BACK')
    elif action_type == 'open_app': # only open the target app # todo::
      app_name = action['app_name']
      self.device.start_app(self.app)
      return
    
    if not event:
      return
    self.device.send_event(event)
    
  def _get_state(self) -> State:
    state = self.device.get_current_state()
    self._state = state
    _, element_list, element_tree = state.text_representation
    return State.create_and_infer_elements(screenshot=state.screenshot_path, element_tree=element_tree)
  
  def _get_stable_state(
      self,
      stability_threshold: int = 3,
      sleep_duration: float = 0.5,
      timeout: float = 6.0,
  ) -> State:
    """Checks if the UI elements remain stable over a number of checks and gets state.

    Args:
        stability_threshold: Number of consecutive checks where UI elements must
          remain the same to consider UI stable.
        sleep_duration: Time in seconds to wait between checks.
        timeout: Maximum time in seconds to wait for UI to become stable before
          giving up.

    Returns:
        True if UI is considered stable, False if it never stabilizes within the
        timeout.
    """
    if not self._prior_state:
      self._prior_state = self._get_state()

    stable_checks = 0
    elapsed_time = 0.0
    current_state = self._get_state()

    while stable_checks < stability_threshold and elapsed_time < timeout:
      if self._prior_state.element_tree.str == current_state.element_tree.str:
        stable_checks += 1
        if stable_checks == stability_threshold:
          break  # Exit early if stability is achieved.
      else:
        stable_checks = 0  # Reset if any change is detected
        self._prior_state = current_state

      time.sleep(sleep_duration)
      elapsed_time += sleep_duration
      current_state = self._get_state()

    return current_state
  
  @property
  def foreground_activity_name(self) -> str:
    return self.state.foreground_activity
  
  @property
  def device_screen_size(self) -> tuple[int, int]:
    return self.state.width, self.state.height

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return self.device_screen_size
  
  def close(self) -> None:
    self.device.disconnect()

  def actions_taken(self) -> list:
    return

  def reset_env(self, wipe_intermidiate_task_data: bool = False) -> State:
    return