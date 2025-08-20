import os
import sys
import time
import logging
import logging
import tools as tools
from agent.environment import AsyncDroidBotEnvForLlamaTouch
from agent.code_agent import CodeAgent
from minimal_setup.config import AgentEnvConfig, LogConfig
from dotenv import load_dotenv

load_dotenv()

def main():
    os.makedirs(LogConfig.LOG_FILE_PATH, exist_ok=True)
    # Setup logging using configuration settings
    log_file_name = f"{LogConfig.LOG_FILE_PATH}/{time.time()}_out.log"
    logging.basicConfig(level=getattr(logging, LogConfig.LOGGING_LEVEL),
                        format=LogConfig.LOGGING_FORMAT,
                        datefmt=LogConfig.LOGGING_DATE_FORMAT,
                        handlers=[logging.FileHandler(log_file_name, 'a'),
                                  logging.StreamHandler()])
    logging.info("Starting environment!")
    agent_env = AsyncDroidBotEnvForLlamaTouch(
        avd_name=AgentEnvConfig.AVD_NAME,
        emulator_controller_args=AgentEnvConfig.EMULATOR_CONTROLLER_AGRS,
        max_steps=AgentEnvConfig.MAX_STEPS,
        local_output_path=AgentEnvConfig.LOCAL_OUTPUT_PATH,
        instruction_fp=AgentEnvConfig.INSTRUCTION_FILE_PATH,
        config=AgentEnvConfig
    )

    results = []
    instruction = None
    recover_from_exception = False
    while True:
      try:
        if not recover_from_exception:
          instruction, app_name = agent_env.get_instruction()
        else:
          recover_from_exception = False

        if instruction is None:
            break
        logging.info(f"Current instruction: {instruction}")

        # initialize agent with appropiate document
        doc_name = f"{AgentEnvConfig.DOCS_BASE_DIR}/{app_name}.json"
        agent_logs_path = f"{agent_env.task_output_path}/agent_logs"

        code_agent = CodeAgent(agent_env, app_name, doc_name, agent_logs_path, AgentEnvConfig.MODEL)

          #### UNCOMMENT FOR DIRECT SCRIPT DEBUGGING 
#         code_agent.FREEZED_CODE = True
#         code_agent.code_config.code = '''
#  $server_overview_screen__you_button.tap()
#  $personal_profile_screen__settings_button.tap()
#  $settings_screen__notifications_button.tap()
# '''


        code_agent.MAX_RETRY_TIMES = 2
        agent_env.reset_env(app_name)
        
        # complete task
        result = code_agent.step(instruction)
        results.append(result)


        # reset environment
      except KeyboardInterrupt:
        logging.info("Keyboard interrupt.")
        agent_env.close()
        sys.exit(0)
      except Exception as e:
        import traceback
        recover_from_exception = True
        traceback.print_exc()

    succeeded = 0
    for res in results:
      if res['is_completed']:
        succeeded += 1
      else:
        print(f'Failed task: \n{instruction}')
    
    tools.dump_json_file(AgentEnvConfig.LOCAL_OUTPUT_PATH,{
        "succeeded":succeeded,
        "results": results
      })
    
    agent_env.close()
    logging.info("Llama Touch Experiment Finished!")
    sys.exit(0)

if __name__ == "__main__":
  main()