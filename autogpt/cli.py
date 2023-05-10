"""Main script for the autogpt package."""
import os
from pathlib import Path
import click
from datetime import datetime 
import shutil
import yaml


@click.group(invoke_without_command=True)
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-settings",
    "-C",
    help="Specifies which ai_settings.yaml file to use, will also automatically skip the re-prompt.",
)
##########
@click.option(
    "--memory-index",
    "memory_index",
    help="Specifies a filepath for which JSON file to use",
)
##########
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "--use-memory",
    "-m",
    "memory_type",
    type=str,
    help="Defines which Memory backend to use",
)
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows Auto-GPT to download files natively.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.pass_context
def main(
    ctx: click.Context,
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    memory_index: str, 
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
) -> None:
    """
    Welcome to AutoGPT an experimental open-source application showcasing the capabilities of the GPT-4 pushing the boundaries of AI.

    Start an Auto-GPT assistant.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    import logging
    import sys

    from colorama import Fore

    from autogpt.agent.agent import Agent
    from autogpt.config import Config, check_openai_api_key
    from autogpt.configurator import create_config
    from autogpt.logs import logger
    from autogpt.memory import get_memory
    from autogpt.prompt import construct_prompt
    from autogpt.utils import get_current_git_branch, get_latest_bulletin

    ####################################
    if memory_index:
        # memory_index_filepath = f"{MEMORY_DIR_PATH}/{memory_index}.json"
        memory_index_filepath = memory_index

    if not os.path.exists(memory_index_filepath):
        os.makedirs(os.path.dirname(memory_index_filepath), exist_ok=True)
        with open(memory_index_filepath, "w") as f:
            f.write("{}") 
        os.chmod(memory_index_filepath, 0o777) # add write permissions 

    ####################################

    # defaults, if ai_settings file doesn't already exist 
    TEMPLATE_MEMORY_PATH = Path(__file__).parent.parent.parent / "memory_files/templates"
    CUSTOM_MEMORY_PATH = Path(__file__).parent.parent.parent / "memory_files/custom_files"
    default_ai_settings_str = "ai_settings_default"
    default_ai_settings_filepath = str((TEMPLATE_MEMORY_PATH / f"{default_ai_settings_str}.yaml").absolute()) 
    
    # if ai_settings filepath str not given, then placeholder: use the datetime instead of client id 
    if not ai_settings:
        datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ai_settings = str((CUSTOM_MEMORY_PATH / f"{default_ai_settings_str}-{datetime}.yaml").absolute())

    # if given but not exists, this is a new session_id session. 
    # Copy the default and create a new file for it. 
    if not os.path.exists(ai_settings):
        os.makedirs(os.path.dirname(ai_settings), exist_ok=True)
        shutil.copyfile(default_ai_settings_filepath, ai_settings)

    ####################################

    if ctx.invoked_subcommand is None:
        cfg = Config()
        # TODO: fill in llm values here
        check_openai_api_key()
        create_config(
            continuous,
            continuous_limit,
            ai_settings,
            skip_reprompt,
            speak,
            debug,
            gpt3only,
            gpt4only,
            memory_type,
            memory_index_filepath,
            browser_name,
            allow_downloads,
            skip_news,
        )
        logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
        ai_name = ""
        if not cfg.skip_news:
            motd = get_latest_bulletin()
            if motd:
                logger.typewriter_log("NEWS: ", Fore.GREEN, motd)
            git_branch = get_current_git_branch()
            if git_branch and git_branch != "stable":
                logger.typewriter_log(
                    "WARNING: ",
                    Fore.RED,
                    f"You are running on `{git_branch}` branch "
                    "- this is not a supported branch.",
                )
            if sys.version_info < (3, 10):
                logger.typewriter_log(
                    "WARNING: ",
                    Fore.RED,
                    "You are running on an older version of Python. "
                    "Some people have observed problems with certain "
                    "parts of Auto-GPT with this version. "
                    "Please consider upgrading to Python 3.10 or higher.",
                )
        system_prompt = construct_prompt()
        # print(prompt)
        # Initialize variables
        full_message_history = []
        next_action_count = 0
        # Make a constant:
        triggering_prompt = (
            "Determine which next command to use, and respond using the"
            " format specified above:"
        )
        # Initialize memory and make sure it is empty.
        # this is particularly important for indexing and referencing pinecone memory
        memory = get_memory(cfg, init=True)
        ##########
        memory.set_memory_filepath(memory_index_filepath)
        ##########
        logger.typewriter_log(
            "Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
        )
        logger.typewriter_log("Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
        agent = Agent(
            ai_name=ai_name,
            memory=memory,
            full_message_history=full_message_history,
            next_action_count=next_action_count,
            system_prompt=system_prompt,
            triggering_prompt=triggering_prompt,
        )
        agent.start_interaction_loop()


if __name__ == "__main__":
    main()
