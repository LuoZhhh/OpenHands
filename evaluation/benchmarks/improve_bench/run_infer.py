"""Implements paper revision based on review comments using OpenHands.

This script allows an LLM agent to:
1. Read the original paper (tex file) and review comments
2. Perform text-based modifications
3. Run additional experiments if needed
4. Output the revised paper as a tex file
"""

import asyncio
import os
from typing import Any

import pandas as pd

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you have completed all revisions, please finish the interaction using the "finish" tool.\n'
}

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'


## TODO: config
def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Get OpenHands configuration for paper revision task."""
    sandbox_config = get_default_sandbox_config_for_eval()
    # Use a container with LaTeX and common ML libraries
    sandbox_config.base_container_image = 'ubuntu:22.04'
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.enable_gpu = True
    sandbox_config.cuda_visible_devices = '0,1'

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=RUN_WITH_BROWSING,
        runtime='docker',
        sandbox_config=sandbox_config,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # get 'draft_editor' config if exists
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = (
        get_llms_for_routing_config()
    )  # Populate with LLMs for routing from config.toml file

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        model_routing=model_routing_config,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    This function:
    1. Sets up the workspace
    2. Installs necessary dependencies (LaTeX, Python libraries)
    3. Clones the paper's GitHub repository
    4. Copies the paper tex file to workspace
    """
    logger.info(f'{"-" * 50} BEGIN Runtime Initialization {"-" * 50}')
    obs: CmdOutputObservation

    # Create workspace
    action = CmdRunAction(command=f"mkdir -p /workspace/{instance['instance_id']}")
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    # Install essential packages
    logger.info('Installing LaTeX and dependencies...')
    action = CmdRunAction(
        command='apt-get update && apt-get install -y texlive-full git python3 python3-pip'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to install latex and dependencies: {str(obs)}',
    )
    # Continue even if some packages fail to install

    # Install Python packages for experiments
    action = CmdRunAction(
        command='pip3 install numpy pandas matplotlib scipy scikit-learn torch torchvision xgboost statsmodels',
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to install Python packages: {str(obs)}',
    )

    # Clone the GitHub repository if provided
    if pd.notna(instance['github_repo']) and instance['github_repo']:
        repo_url = instance['github_repo']
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        action = CmdRunAction(
            command=f'cd /workspace/{instance['instance_id']} && git clone {repo_url}'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to clone repository: {str(obs)}',
        )
        logger.info(f'Successfully cloned repository: {repo_name}')
        instance['repo_name'] = repo_name

    else:
        instance['repo_name'] = None

    # Copy the paper tex file to workspace
    paper_path = instance['paper_path']
    if paper_path and os.path.exists(paper_path):
        action = CmdRunAction(
            command=f'cp {paper_path} /workspace/{instance['instance_id']}/paper.tex'
        )
        action.set_hard_timeout(300)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to copy paper.tex to the workspace: {str(obs)}',
        )
        logger.info(f'Paper tex file copied to /workspace/{instance['instance_id']}/paper.tex')

    # Copy review comments to workspace
    reviews_path = instance.get('reviews_path', '')
    if reviews_path and os.path.exists(reviews_path):
        action = CmdRunAction(
            command=f'cp {reviews_path} /workspace/{instance['instance_id']}/reviews.json'
        )
        action.set_hard_timeout(300)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to copy review.json to the workspace: {str(obs)}',
        )
        logger.info(f'Review comments copied to /workspace/{instance['instance_id']}/reviews.json')

    logger.info(f'{"-" * 50} END Runtime Initialization {"-" * 50}')


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and extract the revised paper.

    This function:
    1. Retrieves the revised paper tex file
    2. Compiles the paper to check for LaTeX errors
    3. Returns the revised paper content
    """
    logger.info(f'{"-" * 50} BEGIN Runtime Completion {"-" * 50}')
    obs: CmdOutputObservation

    # Check if revised paper exists
    action = CmdRunAction(command=f"ls -la /workspace/{instance['instance_id']}")
    action.set_hard_timeout(300)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to show the workspace: {str(obs)}',
    )
    logger.info(f'Workspace contents: {obs.content}')

    # Try to find the revised paper
    # The agent might save it as paper_revised.tex, revised_paper.tex, or overwrite paper.tex
    possible_names = [
        'revised_paper.tex',
    ]

    revised_paper_content = None
    revised_paper_name = None

    for name in possible_names:
        action = CmdRunAction(command=f'test -f /workspace/{instance['instance_id']}/{name} && echo "exists"')
        action.set_hard_timeout(300)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)

        if obs.exit_code == 0 and 'exists' in obs.content:
            action = CmdRunAction(command=f'cat /workspace/{instance['instance_id']}/{name}')
            action.set_hard_timeout(300)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)

            if obs.exit_code == 0:
                revised_paper_content = obs.content
                revised_paper_name = name
                logger.info(f'Found revised paper: {name}')
                break

    # Get any experiment results or logs
    action = CmdRunAction(command=f"find /workspace/{instance['instance_id']} -name '*.log' -o -name '*.txt' -o -name '*.csv' | head -20")
    action.set_hard_timeout(300)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    output_files = obs.content if obs.exit_code == 0 else ''

    logger.info(f'{"-" * 50} END Runtime Completion {"-" * 50}')

    return {
        'revised_paper_content': revised_paper_content,
        'revised_paper_name': revised_paper_name,
        'output_files': output_files,
    }


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True
):
    """Process a single paper revision instance."""
    config = get_config(metadata)

    # Setup the logger
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance['instance_id'], log_dir)
    else:
        logger.info(f'Starting revision for paper {instance["instance_id"]}.')

    # Prepare the instruction for the agent
    instruction = f"""You are tasked with revising a research paper based on reviewer comments.

Paper Information:
- Paper ID: {instance['instance_id']}
- Paper Title: {instance.get('title', 'N/A')}
- Paper Location: /workspace/{instance['instance_id']}/paper.tex

Review Comments Location: /workspace/{instance['instance_id']}/reviews.json
"""

    if instance.get('repo_name'):
        instruction += f"""
GitHub Repository: /workspace/{instance['instance_id']}/{instance['repo_name']}
You can access the code, run experiments, and add new results as needed.
"""

    instruction += f"""
Your Tasks:
1. Read the original paper (/workspace/{instance['instance_id']}/paper.tex) and review comments (/workspace/{instance['instance_id']}/reviews.txt)
2. Address each review comment by:
   - Making text modifications directly to the paper (tex file)
   - Running additional experiments if requested (based on the provided code repository)
   - Incorporating the additional experimental results to the paper (tex file)
3. Save the revised paper as /workspace/{instance['instance_id']}/revised_paper.tex

Guidelines:
- Maintain the paper's LaTeX formatting and structure
- Be thorough in addressing all reviewer concerns
- If experiments are needed, run them and integrate results into the paper
- Keep track of all changes made
- Ensure the revised paper compiles without errors

"""
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

    # Create runtime and run the agent
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance)

    # Run the controller
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                metadata.agent_class
            ),
        )
    )

    if state is None:
        raise ValueError('State should not be None.')

    metrics = get_metrics(state)

    # Get the revised paper
    result = complete_runtime(runtime, instance)

    # Save the revised paper to output directory
    if result['revised_paper_content']:
        output_paper_path = os.path.join(
            metadata.eval_output_dir,
            'revised_papers',
            f"{instance['instance_id']}_revised.tex"
        )
        os.makedirs(os.path.dirname(output_paper_path), exist_ok=True)

        with open(output_paper_path, 'w') as f:
            f.write(result['revised_paper_content'])

        logger.info(f'Revised paper saved to: {output_paper_path}')
        result['output_paper_path'] = output_paper_path
    else:
        logger.error('No revised paper found!')
        result['output_paper_path'] = None

    # Convert history for compatibility
    histories = compatibility_for_eval_history_pairs(state.history)

    # Save the output
    output = EvalOutput(
        instance_id=instance['instance_id'],
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        test_result=result,
        error=state.last_error if state and state.last_error else None,
    )

    return output


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--papers-csv',
        type=str,
        required=True,
        help='CSV file containing paper information (columns: instance_id, title, paper_path, reviews_path, repo_path)'
    )
    args, _ = parser.parse_known_args()

    # Load the dataset
    dataset = pd.read_csv(args.papers_csv)

    # Validate required columns
    required_columns = ['instance_id', 'paper_path', 'reviews_path']
    for col in required_columns:
        if col not in dataset.columns:
            raise ValueError(f'Dataset must contain column: {col}')

    # Get LLM config
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Create metadata
    metadata = make_metadata(
        llm_config,
        'paper-revision',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    # Run the evaluation
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )

    print(f'\n{"=" * 60}')
    print(f'Paper Revision Complete!')
    print(f'Revised papers saved to: {os.path.join(metadata.eval_output_dir, "revised_papers")}')
    print(f'Output log saved to: {output_file}')
    print(f'{"=" * 60}\n')
