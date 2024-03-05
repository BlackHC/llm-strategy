"""
Spike for a meta-loop that optimizes the prompt templates we use.
"""
# flake8: noqa
import traceback
from datetime import datetime
from typing import Generic, TypeVar

import langchain
import wandb
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import OutputParserException
from llmtracer import JsonFileWriter, TraceViewerIntegration, wandb_tracer
from openai import OpenAIError
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from rich.console import Console
from rich.traceback import install

from llm_hyperparameters.track_execution import (
    LangchainInterface,
    TrackedChatModel,
    get_tracked_chats,
    track_langchain,
)
from llm_hyperparameters.track_hyperparameters import (
    Hyperparameters,
    hyperparameters_scope,
    track_hyperparameters,
)
from llm_strategy.chat_chain import ChatChain
from llm_strategy.llm_function import LLMFunction, llm_explicit_function, llm_function

install(show_locals=True, width=190, console=Console(width=190, color_system="truecolor", stderr=True))


langchain.llm_cache = SQLiteCache(".optimization_unit.langchain.db")

chat_model = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    request_timeout=500,
    max_tokens=4096,
    temperature=0.5,
)
chat_model = TrackedChatModel(chat_model=chat_model)

simpler_chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    request_timeout=240,
    max_tokens=2048,
    temperature=0.5,
)
simple_chat_model = TrackedChatModel(chat_model=simpler_chat_model)
# %%

T_TaskParameters = TypeVar("T_TaskParameters")
T_TaskResults = TypeVar("T_TaskResults")
T_Hyperparameters = TypeVar("T_Hyperparameters")


class TaskRun(GenericModel, Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The task run. This is the 'data' we use to optimize the hyperparameters.
    """

    task_parameters: T_TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: T_Hyperparameters = Field(
        ...,
        description="The hyperparameters used for the task. We optimize these.",
    )
    all_chat_chains: dict = Field(..., description="The chat chains from the task execution.")
    return_value: T_TaskResults | None = Field(
        ..., description="The results of the task. (None for exceptions/failure.)"
    )
    exception: list[str] | str | None = Field(..., description="Exception that occurred during the task execution.")


class TaskReflection(BaseModel):
    """
    The reflections on the task.

    This contains the lessons we learn from each task run to come up with better
    hyperparameters to try.
    """

    feedback: str = Field(
        ...,
        description=(
            "Only look at the final results field. Does its content satisfy the "
            "task description and task parameters? Does it contain all the relevant "
            "information from the all_chains and all_prompts fields? What could be improved "
            "in the results?"
        ),
    )
    evaluation: str = Field(
        ...,
        description=(
            "The evaluation of the outputs given the task. Is the output satisfying? What is wrong? What is missing?"
        ),
    )
    hyperparameter_suggestion: str = Field(
        ...,
        description="How we want to change the hyperparameters to improve the results. What could we try to change?",
    )
    hyperparameter_missing: str = Field(
        ...,
        description=(
            "What hyperparameters are missing to improve the results? What could "
            "be changed that is not exposed via hyperparameters?"
        ),
    )


class TaskInfo(GenericModel, Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The task run and the reflection on the experiment.
    """

    task_parameters: T_TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: T_Hyperparameters = Field(
        ...,
        description="The hyperparameters used for the task. We optimize these.",
    )
    reflection: TaskReflection = Field(..., description="The reflection on the task.")


class OptimizationInfo(GenericModel, Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The optimization information. This is the data we use to optimize the
    hyperparameters.
    """

    older_task_summary: str | None = Field(
        None,
        description=(
            "A summary of previous experiments and the proposed changes with "
            "the goal of avoiding trying the same changes repeatedly."
        ),
    )
    task_infos: list[TaskInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]] = Field(
        ..., description="The most recent tasks we have run and our reflections on them."
    )
    best_hyperparameters: T_Hyperparameters = Field(..., description="The best hyperparameters we have found so far.")


class OptimizationStep(GenericModel, Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The next optimization steps. New hyperparameters we want to try experiments and new
    task parameters we want to evaluate on given the previous experiments.
    """

    best_hyperparameters: T_Hyperparameters = Field(
        ...,
        description="The best hyperparameters we have found so far given task_infos and history.",
    )
    suggestion: str = Field(
        ...,
        description=(
            "The suggestions for the next experiments. What could we try to "
            "change? We will try several tasks next and several sets of hyperparameters. "
            "Let's think step by step."
        ),
    )
    task_parameters_suggestions: list[T_TaskParameters] = Field(
        ...,
        description="The task parameters we want to try next.",
        hint_min_items=1,
        hint_max_items=4,
    )
    hyperparameter_suggestions: list[T_Hyperparameters] = Field(
        ...,
        description="The hyperparameters we want to try next.",
        hint_min_items=1,
        hint_max_items=2,
    )


class ImprovementProbability(BaseModel):
    considerations: list[str] = Field(..., description="The considerations for potential improvements.")
    probability: float = Field(..., description="The probability of improvement.")


class LLMOptimizer:
    @llm_explicit_function
    @staticmethod
    def reflect_on_task_run(
        language_model,
        task_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters],
    ) -> TaskReflection:
        """
        Reflect on the results given the task parameters and hyperparameters.

        This contains the lessons we learn from each task run to come up with better
        hyperparameters to try.
        """
        raise NotImplementedError()

    @llm_explicit_function
    @staticmethod
    def summarize_optimization_info(
        language_model,
        optimization_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters],
    ) -> str:
        """
        Summarize the optimization info. We want to preserve all relevant knowledge for
        improving the hyperparameters in the future. All information from previous
        experiments will be forgotten except for what this summary.
        """
        raise NotImplementedError()

    @llm_explicit_function
    @staticmethod
    def suggest_next_optimization_step(
        language_model,
        optimization_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters],
    ) -> OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
        """
        Suggest the next optimization step.
        """
        raise NotImplementedError()

    @llm_explicit_function
    @staticmethod
    def probability_for_improvement(
        language_model,
        optimization_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters],
    ) -> ImprovementProbability:
        """
        Return the probability for improvement (between 0 and 1).

        This is your confidence that your next optimization steps will improve the
        hyperparameters given the information provided. If you think that the
        information available is unlikely to lead to better hyperparameters, return 0.
        If you think that the information available is very likely to lead to better
        hyperparameters, return 1. Be concise.
        """
        raise NotImplementedError()


def capture_task_run(
    llm_interface: LangchainInterface,
    task_executor: LLMFunction,
    task_parameters: T_TaskParameters,
    hyperparameters: BaseModel,
) -> TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
    """Capture the task run."""
    tracked_chat_model = track_langchain(llm_interface)

    # TODO: add a way to ignore the hyperparameter warnings
    structured_prompt = task_executor.llm_bound_signature(**dict(task_parameters)).structured_prompt

    exception = None
    return_value = None
    with hyperparameters_scope(hyperparameters) as scope:
        try:
            return_value = task_executor.explicit(tracked_chat_model, task_parameters)
        except (OpenAIError, OutputParserException) as e:
            # convert the exception to a string and include a bit of context
            # so we can debug it later
            exception = traceback.format_exception_only(e)

    print(return_value)

    all_chat_chains = get_tracked_chats(tracked_chat_model)
    return TaskRun[structured_prompt.input_type, structured_prompt.return_annotation, type(scope.hyperparameters)](
        task_parameters=task_parameters,
        hyperparameters=scope.hyperparameters,
        all_chat_chains=all_chat_chains,
        return_value=return_value,
        exception=exception,
    )


@track_hyperparameters
def optimize_hyperparameters(
    chat_model: BaseChatModel,
    task_chat_model: BaseChatModel,
    task_executor,
    seed_task_parameters: list[T_TaskParameters],
) -> OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
    """Optimize the hyperparameters."""
    task_root_chain = ChatChain(task_chat_model, [])

    seed_task_runs = [
        capture_task_run(
            llm_interface=task_root_chain,
            task_executor=task_executor,
            task_parameters=task_parameters,
            hyperparameters=BaseModel(),
        )
        for task_parameters in seed_task_parameters
    ]

    root_chain = ChatChain(chat_model, [])
    llm_bound_signature = task_executor.llm_bound_signature(**dict(seed_task_runs[0].task_parameters))

    task_infos = []
    for task_run in seed_task_runs:
        task_infos.append(
            TaskInfo[
                llm_bound_signature.input_type,
                llm_bound_signature.return_annotation,
                type(task_run.hyperparameters),
            ](
                task_parameters=task_run.task_parameters,
                hyperparameters=task_run.hyperparameters,
                results=task_run.return_value,
                reflection=LLMOptimizer.reflect_on_task_run(root_chain, task_run),
            )
        )

    hyperparamters = Hyperparameters.merge(task_info.hyperparameters for task_info in task_infos)
    hyperparameter_type = type(hyperparamters)

    optimization_info = OptimizationInfo[
        llm_bound_signature.input_type,
        llm_bound_signature.return_annotation,
        hyperparameter_type,
    ](
        task_infos=task_infos,
        best_hyperparameters=hyperparamters,
    )

    optimization_step = None
    for _ in range(2):
        optimization_step = LLMOptimizer.suggest_next_optimization_step(root_chain, optimization_info)

        optimization_info.best_hyperparameters = optimization_step.best_hyperparameters

        for task_parameters in optimization_step.task_parameters_suggestions:
            for hyperparameters in optimization_step.hyperparameter_suggestions:
                task_run = capture_task_run(task_root_chain, task_executor, task_parameters, hyperparameters)
                task_info = TaskInfo[
                    llm_bound_signature.input_type,
                    llm_bound_signature.return_annotation,
                    type(hyperparameters),
                ](
                    task_parameters=task_run.task_parameters,
                    hyperparameters=task_run.hyperparameters,
                    results=task_run.return_value,
                    reflection=LLMOptimizer.reflect_on_task_run(root_chain, task_run),
                )
                optimization_info.task_infos.append(task_info)

        if len(optimization_info.task_infos) >= 20:
            optimization_info.older_task_summary = LLMOptimizer.summarize_optimization_info(
                root_chain,
                OptimizationInfo[
                    llm_bound_signature.input_type,
                    llm_bound_signature.return_annotation,
                    hyperparameter_type,
                ](
                    older_task_summary=optimization_info.older_task_summary,
                    task_infos=optimization_info.task_infos[:-10],
                    best_hyperparameters=optimization_step.best_hyperparameters,
                ),
            )
            optimization_info.task_infos = optimization_info.task_infos[-10:]

        if LLMOptimizer.probability_for_improvement(root_chain, optimization_info).probability < 0.5:
            break

    return optimization_step


@llm_function
def write_essay(chat_model, essay_topic: str = Field(..., description="The essay topic.")) -> str:
    """
    Write an essay at the level of an Oxford undergraduate student. Please write about 500 words.
    Use markdown to format the essay.
    """
    raise NotImplementedError()


@llm_function
def create_essay_topics(
    chat_model,
    domain: str = Field(..., description="The domain or general area of the essay topic"),
    n: int = Field(..., description="Number of topics to generate"),
) -> list[str]:
    """Create a list of essay topics."""
    raise NotImplementedError()


# Several very varied essay topics in various domains (e.g., science, technology, art, philosophy, etc.).
essay_topics = [
    "The Interplay Between Artificial Intelligence and Human Ethics: A Philosophical Inquiry",
    "Examining the Effects of Climate Change on Global Food Security and Agricultural Practices",
    "A Comparative Analysis of Ancient Greek and Roman Political Structures: Democracy vs. Republic",
    "The Psychological Impact of Social Media: A Deep Dive into Mental Health and Connectivity",
    "The Fusion of Quantum Mechanics and General Relativity: The Road Towards a Unified Theory",
    "Exploring Gender Roles in Shakespeare's Plays: A Cross-Sectional Study of Female Characters",
    "The Evolution of Human Language: Examining Linguistic Diversity and its Implications",
    "The Intersection of Neuroscience and Music: How Rhythm and Melody Influence Cognitive Function",
    "The Socioeconomic Impacts of Renewable Energy Adoption: A Comparative Study of Developed and Developing Nations",
    "Deconstructing the Notion of Free Will: A Multidisciplinary Analysis of Human Agency",
]

# %%

wandb.init(project="blackboard-pagi", name="optimization_unit_spike")


def get_json_trace_filename(title: str) -> str:
    """Get a filename for the resulting trace based on the title and the current date+time."""
    return f"{title}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"


event_handlers = [
    JsonFileWriter(get_json_trace_filename("optimization_unit_trace")),
    TraceViewerIntegration(),
]


with wandb_tracer(
    "BBO",
    module_filters="__main__",
    stack_frame_context=0,
    event_handlers=event_handlers,
) as trace_builder:
    # seed_task_parameters = [
    #     create_essay_topics.get_input_object(domain="comedy", n=2),
    #     create_essay_topics.get_input_object(domain="drama", n=2),
    # ]

    seed_task_parameters = [
        write_essay.get_input_object(
            essay_topic="Examining the Effects of Climate Change on Global Food Security and Agricultural Practices"
        ),
        write_essay.get_input_object(
            essay_topic="The Fusion of Quantum Mechanics and General Relativity: The Road Towards a Unified Theory"
        ),
    ]

    optimization_step = optimize_hyperparameters(chat_model, simpler_chat_model, write_essay, seed_task_parameters)
