import json
import logging
import os
import re
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

from .common.action_type import Action, ActionType


class Agent(Enum):
    MIND2WEB = "mind2web"
    SEECLICK = "seeclick"
    AUTODROID = "autodroid"
    AUTODROID_V2 = "autodroidv2"
    COGAGENT = "cogagent"


class TaskCategory(Enum):
    GENERAL = "general"
    GOOGLEAPPS = "googleapps"
    INSTALL = "install"
    WEBSHOPPING = "webshopping"
    GENERATED = "generated"


ACTION_SPACE = {
    "Home": ActionType.PRESS_HOME,
    "Back": ActionType.PRESS_BACK,
    "Click": ActionType.DUAL_POINT,
    "Swipe": ActionType.DUAL_POINT,
    "Input": ActionType.TYPE,
}


class EssentialStateKeyword(Enum):
    """fuzzy match item"""

    FUZZY = "fuzzy"

    """exact UI component match"""
    EXACT = "exact"
    EXCLUDE = "exclude"

    """exact activity match"""
    ACTIVITY = "activity"

    """exact action match"""
    CLICK = "click"
    TYPE = "type"

    """exact system state match"""
    CHECK_INSTALL = "check_install"
    CHECK_UNINSTALL = "check_uninstall"


class UIState:
    """
    - index: int, index of the UIState in a trace
    - screenshot_path: string
    - vh_path: string, dumped through `uiautomator`
    - vh_json_path: string, dumped through `droidbot`
    - vh_simp_ui_json_path: string|None, only exist in the ground-truth dataset;
        used to visualize important UI components
    - activity: stirng, activity of the current screen
    - action: Action
    - state_type: string ["groundtruth", "execution"], type of the UIState
    - essential_state: Dict[
        "fuzzy": ["0", "1"],
        "check_install": ["Microsoft Excel"],
        ...
        ]
    """

    def __init__(
        self,
        index: int,
        screenshot_path: str,
        vh_path: str,
        vh_json_path: str,
        activity: str,
        action: Action,
        state_type: str,
        vh_simp_ui_json_path: Optional[str] = None,  # only gr-trace contains this field
    ) -> None:
        assert type(index) == int
        self.index: int = index

        assert state_type in ["groundtruth", "execution"]
        self.state_type: str = state_type

        self.screenshot_path: str = screenshot_path
        self.vh_path: str = vh_path
        self.vh_json_path = vh_json_path
        self.vh_simp_ui_json_path: str = vh_simp_ui_json_path
        self.activity: str = activity
        self.action: Action = action

        # load annotated essential_state if it is ground-truth UIState
        self.essential_state: Optional[
            DefaultDict[EssentialStateKeyword, List[str]]
        ] = None
        if self.state_type == "groundtruth":
            assert self.vh_simp_ui_json_path is not None
            # check whether this UIState has annotated essential states (file
            # postfix: .ess). if so, load the essential states
            potential_es_file = self.screenshot_path.replace(".png", ".ess")
            if os.path.exists(potential_es_file):
                with open(potential_es_file, "r") as f:
                    content = f.read()
                self.essential_state = DefaultDict(list)
                # split_content: ['exact<1>',
                #                 'fuzzy<-1>',
                #                 'check_install<Microsoft Excel>',
                #                  ...]
                split_content = [item.strip() for item in content.split("|")]
                for item in split_content:
                    match = re.search(r"(?P<keyword>\w+)<(?P<content>.+)>", item)
                    if match:
                        keyword: str = match.group("keyword")
                        content: str = match.group("content")
                        self.essential_state[
                            EssentialStateKeyword[keyword.upper()]
                        ].append(content)
        elif self.state_type == "execution":
            assert self.vh_simp_ui_json_path is None
        else:
            pass

        self.installed_app_path: Optional[str] = None
        if self.state_type == "execution":
            self.installed_app_path = os.path.join(
                self.screenshot_path.split("screenshot")[0],
                "installed_apps",
                "installed_apps.txt",
            )

    def get_bbox_bounds_by_keyword_id(self, keyword_id: int) -> Tuple[float]:
        """
        Get the bounding box of the keyword_id-th essential state
        """
        assert self.essential_state is not None

        data: List[Dict] = json.load(open(self.vh_simp_ui_json_path, "r"))
        bounds: str = data[keyword_id]["bounds"]
        left, top, right, bottom = map(
            float, re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)[0]
        )
        return left, top, right, bottom


TaskTrace = List[UIState]


def get_all_screenshot_paths(task_trace: TaskTrace) -> List[str]:
    return [ui_state.screenshot_path for ui_state in task_trace]


def get_all_vh_paths(task_trace: TaskTrace) -> List[str]:
    return [ui_state.vh_path for ui_state in task_trace]


def get_all_actions(task_trace: TaskTrace) -> List[Action]:
    return [ui_state.action for ui_state in task_trace]


class DatasetHelper:
    """A singleton class to help load task metadata from the our constructed dataset."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatasetHelper, cls).__new__(cls)
        return cls._instance

    def __init__(self, epi_metadata_path: str, gr_dataset_path: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.epi_metadata_path = epi_metadata_path
        self.gr_dataset_path = gr_dataset_path
        self.logger.info(
            f"Using epi_metadata_path: {self.epi_metadata_path}, gr_dataset_path: {gr_dataset_path}"
        )

        """Example of epi_metadata_dict: 
        {
            "episode": {
                "category": TaskCategory,
                "task_description": str,
            },
            ...
        }
        this dict will only be loaded when *self.epi_metadata_dict* is accessed
        """
        self._epi_metadata_dict: OrderedDict[
            str, Dict[str, Union[TaskCategory, str]]
        ] = OrderedDict()

    @property
    def epi_metadata_dict(self):
        if not self._epi_metadata_dict:
            self.init_epi_to_category()
        return self._epi_metadata_dict

    def init_epi_to_category(self) -> None:
        """Load episode metadata from the file {self.epi_metadata_path}
        Columns: [episode,category,path,description,nsteps]

        Format: {
            "episode": {
                "category": TaskCategory,
                "task_description": str,
            },
            ...
        }
        """
        assert os.path.exists(
            self.epi_metadata_path
        ), f"The file {self.epi_metadata_path} does not exist"

        with open(self.epi_metadata_path, "r") as f:
            next(f)  # f is an iterable file object; skip the header
            for line in f:
                items = line.strip().split("\t")
                app = ""
                if len(items) < 6:
                    epi, category, path, task_description, steps = items[:4]
                else:
                    epi, category, path, task_description, steps, app = items[:6]
                self._epi_metadata_dict[epi] = {
                    # convert string-format category to TaskCategory,
                    # e.g., "general" -> TaskCategory.GENERAL
                    "category": TaskCategory[category.strip().upper()],
                    "task_description": task_description,
                    "path": path,
                    "app": app,
                    "steps": steps,
                }

    def get_all_episodes(self) -> List[str]:
        return [*self.epi_metadata_dict.keys()]

    def get_episodes_by_category(self, episode: Union[TaskCategory, str]) -> List[str]:
        if isinstance(episode, str):
            episode = TaskCategory[episode.upper()]
        return [
            epi
            for epi in self.get_all_episodes()
            if self.get_category_by_episode(epi) == episode
        ]

    def get_task_description_by_episode(self, episode) -> str:
        if episode not in self.epi_metadata_dict:
            raise KeyError(f"episode: {episode} not found in dataset")
        return self.epi_metadata_dict[episode]["task_description"]
    
    def get_task_steps_by_episode(self, episode) -> str:
        if episode not in self.epi_metadata_dict:
            raise KeyError(f"episode: {episode} not found in dataset")
        return self.epi_metadata_dict[episode]["steps"]
    
    def get_task_app_by_episode(self, episode) -> str:
        if episode not in self.epi_metadata_dict:
            raise KeyError(f"episode: {episode} not found in dataset")
        return self.epi_metadata_dict[episode]["app"]
    
    def get_task_path_by_episode(self, episode) -> str:
        if episode not in self.epi_metadata_dict:
            raise KeyError(f"episode: {episode} not found in dataset")
        return self.epi_metadata_dict[episode]["path"]

    def get_category_by_episode(self, episode) -> TaskCategory:
        if episode not in self.epi_metadata_dict:
            raise KeyError(f"episode: {episode} not found in dataset")
        return self.epi_metadata_dict[episode]["category"]

    # ---------------------------------------------------- #
    # -------- Processing testbed exectuion traces ------- #
    # -------- Used for the testbed evaluator ------------ #
    # -- Methods                                           #
    # ---- load_testbed_trace_by_episode                   #
    # ---- load_testbed_trace_by_path                      #
    # -- Data (always in the captured_data folder)         #
    # ---- xml: [0.xml, 1.xml, ...]                        #
    # ---- activity: [0.activity, 1.activity, ...]         #
    # ---- action: [-1.action, 0.action, 1.action, ...]    #
    # ---- screenshot: [0.png, 1.png, ...]                 #
    # ---------------------------------------------------- #
    def _proc_testbed_trace_action_file(self, action_file) -> Action:
        """
        action_type:
            - "CLICK"
            - "SWIPE"
            - "TYPE"
            - "PRESS_BACK"
            - "PRESS_HOME"
            - "PRESS_ENTER"
            - "STATUS_TASK_COMPLETE"
            - "STATUS_TASK_IMPOSSIBLE"

        action_param:
            - "CLICK": [x, y]
            - "SWIPE": [st_x, st_y, end_x, end_y]
            - "TYPE": str
            - others: None

        examples:
            - "TYPE|good burger place|NULL|1080|2400"
            - "CLICK|[0.0879 0.9069]|NULL|1080|2400"
            - "CLICK|[0.0879, 0.9069]|NULL|1080|2400"
            - "SWIPE|[0.8 0.5]|[0.2 0.5]|1080|2400"
            - "SWIPE|[0.8, 0.5]|[0.2 0.5]|1080|2400"
            - "PRESS_HOME|NULL|NULL|1080|2400"
        """
        with open(action_file) as f:
            action_repr = f.read()
        action_repr = action_repr.split("|")
        action_type = action_repr[0]
        if action_repr[2] != "NULL":
            pattern = r"\[(-?\d+\.\d+),?\s+(-?\d+\.\d+)\]"
            x1, y1 = re.search(pattern, action_repr[1]).groups()
            x2, y2 = re.search(pattern, action_repr[2]).groups()
            action_param = [float(x1), float(y1), float(x2), float(y2)]
        elif action_repr[1] != "NULL" and (
            action_type == "CLICK" or action_type == "SWIPE"
        ):
            pattern = r"\[\s*(-?\d+\.\d+),?\s+(-?\d+\.\d+)\s*\]"
            x1, y1 = re.search(pattern, action_repr[1]).groups()
            action_param = [float(x1), float(y1)]
        elif action_type == "TYPE":
            action_param = action_repr[1]
        else:
            action_param = None
        screen_width = int(action_repr[-2])
        screen_height = int(action_repr[-1])

        typed_text = ""
        touch_point_yx = lift_point_yx = (-1, -1)

        if action_type == "SWIPE":
            action_type = "DUAL_POINT"
            touch_point_yx = (action_param[1], action_param[0])
            lift_point_yx = (action_param[3], action_param[2])
        elif action_type == "CLICK":
            action_type = "DUAL_POINT"
            touch_point_yx = lift_point_yx = (action_param[1], action_param[0])
        elif action_type == "TYPE":
            typed_text = action_param[0]
        action = Action(
            action_type=ActionType[action_type.upper()],
            touch_point_yx=touch_point_yx,
            lift_point_yx=lift_point_yx,
            typed_text=typed_text,
        )

        return action

    def load_testbed_trace_by_path(self, path: str) -> TaskTrace:
        screenshot_folder_path = os.path.join(path, "screenshot")
        num_UIState = len(os.listdir(screenshot_folder_path))
        task_trace: List[UIState] = []
        for i in range(num_UIState):
            screenshot_path = os.path.join(screenshot_folder_path, f"{i}.png")
            xml_path = os.path.join(path, "xml", f"{i}.xml")
            vh_json_path = os.path.join(path, "view_hierarchy", f"{i}.json")

            # activity_path = os.path.join(path, "activity", f"{i}.activity")
            # activity = self._extract_activity_from_file(activity_path)
            activity = None
            action_path = os.path.join(path, "action", f"{i}.action")
            if not os.path.exists(action_path):
                action = None
            else:
                action = self._proc_testbed_trace_action_file(action_path)

            ui_state = UIState(
                index=i,
                screenshot_path=screenshot_path,
                vh_path=xml_path,
                vh_json_path=vh_json_path,
                activity=activity,
                action=action,
                state_type="execution",
            )
            task_trace.append(ui_state)
        return task_trace

    # ---------------------------------------------------- #
    # -- Processing the ground-truth trace we annotated -- #
    # -- Used for the exact evaluator and task execution - #
    # -- Exposed methods                                   #
    # ---- load_groundtruth_trace_by_episode               #
    # ---------------------------------------------------- #
    def load_groundtruth_trace_by_episode(self, episode: str) -> Optional[TaskTrace]:
        category: TaskCategory = self.get_category_by_episode(episode)
        # self.logger.info(f"episode: {episode}, category: {category}")
        if episode in self._load_groundtruth_trace_by_category(category):
            return self._load_groundtruth_trace_by_category(category)[episode]
        else:
            return None

    @lru_cache(maxsize=None)
    def _load_groundtruth_trace_by_category(
        self, category: TaskCategory
    ) -> Dict[str, TaskTrace]:
        """
        Load ground-truth traces in a whole category
        *Note*: There is a potential risk when directly invoking this method, as
        the ground-truth trace dict may contain traces that their episodes are not
        in self.epi_metadata_dict. Always loading ground-truth traces by episode.


        Return: {
            "episode_id_1": [(screenshot_1_1, XML_1_1, action_1_1), (screenshot_1_2, XML_1_2, action_1_2), ...],
            "episode_id_2": [(screenshot_2_1, XML_2_1, action_2_1), (screenshot_2_2, XML_2_2, action_2_2), ...],
            ...
        }
        """
        gr_category_path = os.path.join(self.gr_dataset_path, category.value)
        gt_trace_dict = {}
        dirs = [
            d
            for d in os.listdir(gr_category_path)
            if os.path.isdir(os.path.join(gr_category_path, d))
        ]
        dirs.sort()
        for dir in dirs:
            path = os.path.join(gr_category_path, dir)
            ep_id_path = os.path.join(path, "instruction.txt")
            with open(ep_id_path, "r") as f:
                ep_id = f.readline().strip()

            ep_trace: TaskTrace = self._load_groundtruth_trace_by_path(path)
            gt_trace_dict[ep_id] = ep_trace

        return gt_trace_dict

    def _extract_actions_from_file(self, path: str) -> List[Action]:
        """Actions for one episode are recorded in one file.
        Format:
        [Home]
        [Click] Screen Resolution (320, 720), Click Position (176, 564)
        [Click] Screen Resolution (320, 720), Click Position (112, 46)
        [Input] bestbuy.com
        [Click] Screen Resolution (320, 720), Click Position (100, 83)
        [Click] Screen Resolution (320, 720), Click Position (158, 232)
        [Input] best rated video games
        [Swipe] Screen Resolution (320, 720), Start Position (153, 664), End Position (164, 69)
        [Click] Screen Resolution (320, 720), Click Position (275, 545)
        ...
        """
        action_list = []
        action_texts = open(path, "r").readlines()

        # this for-range is for processing the action record
        for action_text in action_texts:
            action_type = re.search(
                r"\[(?P<action_type>.+)\]", action_text
            ).groupdict()["action_type"]
            if action_type == "Home" or action_type == "Back":
                action_list.append(Action(action_type=ACTION_SPACE[action_type]))
            elif action_type == "Click":
                pattern = re.compile(
                    r"Screen Resolution \((?P<screen_width>\d+), (?P<screen_height>\d+)\), Click Position \((?P<position_1_x>\d+), (?P<position_1_y>\d+)\)"
                )
                re_dict = re.search(pattern, action_text).groupdict()
                screen_width = int(re_dict["screen_width"])
                screen_height = int(re_dict["screen_height"])
                action_list.append(
                    Action(
                        action_type=ACTION_SPACE[action_type],
                        touch_point_yx=(
                            int(re_dict["position_1_y"]) / screen_height,
                            int(re_dict["position_1_x"]) / screen_width,
                        ),
                        lift_point_yx=(
                            int(re_dict["position_1_y"]) / screen_height,
                            int(re_dict["position_1_x"]) / screen_width,
                        ),
                    )
                )
            elif action_type == "Swipe":
                pattern = re.compile(
                    r"Screen Resolution \((?P<screen_width>\d+), (?P<screen_height>\d+)\), Start Position \((?P<position_1_x>\d+), (?P<position_1_y>\d+)\), End Position \((?P<position_2_x>\d+), (?P<position_2_y>\d+)\)"
                )
                re_dict = re.search(pattern, action_text).groupdict()
                screen_width = int(re_dict["screen_width"])
                screen_height = int(re_dict["screen_height"])
                action_list.append(
                    Action(
                        action_type=ACTION_SPACE[action_type],
                        touch_point_yx=(
                            int(re_dict["position_1_y"]) / screen_height,
                            int(re_dict["position_1_x"]) / screen_width,
                        ),
                        lift_point_yx=(
                            int(re_dict["position_2_y"]) / screen_height,
                            int(re_dict["position_2_x"]) / screen_width,
                        ),
                    )
                )
            elif action_type == "Input":
                pattern = re.compile(r"\[Input\] (?P<text>.*)")
                text = re.search(pattern, action_text).groupdict()["text"]
                action_list.append(
                    Action(action_type=ACTION_SPACE[action_type], typed_text=text)
                )
            else:
                raise ValueError(f"Unknown action type: {action_type}")

        # At the end of list, add one TASK_COMPLETE Action as this is missing in
        # the *eventStructure.txt* file.
        action_list.append(Action(action_type=ActionType.STATUS_TASK_COMPLETE))

        return action_list

    def _extract_activity_from_file(self, path: str) -> str:
        """convert com.android.settings/.Settings to com.android.settings.Settings"""
        with open(path) as f:
            line = f.read().strip()
        if "mObscuringWindow" in line:
            raise Exception(f"Activity format error: {line}")

        if "/." in line:
            line = line.replace("/.", ".")

        return line

    def _load_groundtruth_trace_by_path(self, path: str) -> TaskTrace:
        self.logger.debug(f"loading groundtruth trace in path: {path}")
        ep_trace_list: TaskTrace = []
        # the task trace folder may contain png-format images
        # their name could be 0.png, 1.png, and 0_drawed.png, 1_drawed.png
        # 0_drawed.png and 1_drawed.png are used in the annotation process
        # we only want to get 0.png and 1.png
        files = [
            f for f in os.listdir(path) if f.endswith(".png") and "drawed" not in f
        ]
        files.sort()

        action_path = os.path.join(path, "eventStructs.txt")
        action_list = self._extract_actions_from_file(action_path)

        # iterate in [0, 1, 2, 3, ..., # of all ui states]
        for i in range(len(action_list)):
            action = action_list[i]
            img_path = os.path.join(path, f"{i}.png")
            xml_path = img_path.replace("png", "xml")
            vh_json_path = img_path.replace("png", "vh")
            vh_simp_ui_json_path = img_path.replace("png", "json")
            activity_file = img_path.replace("png", "activity")
            activity = self._extract_activity_from_file(activity_file)
            ep_trace_list.append(
                UIState(
                    index=i,
                    screenshot_path=img_path,
                    vh_path=xml_path,
                    vh_json_path=vh_json_path,
                    vh_simp_ui_json_path=vh_simp_ui_json_path,
                    activity=activity,
                    action=action,
                    state_type="groundtruth",
                )
            )
        return ep_trace_list
