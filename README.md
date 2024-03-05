# llm-strategy

[![Release](https://img.shields.io/github/v/release/blackhc/llm-strategy)](https://img.shields.io/github/v/release/blackhc/llm-strategy)
[![Build status](https://img.shields.io/github/actions/workflow/status/blackhc/llm-strategy/main.yml?branch=main)](https://github.com/blackhc/llm-strategy/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/blackhc/llm-strategy/branch/main/graph/badge.svg)](https://codecov.io/gh/blackhc/llm-strategy)
[![Commit activity](https://img.shields.io/github/commit-activity/m/blackhc/llm-strategy)](https://img.shields.io/github/commit-activity/m/blackhc/llm-strategy)
[![License](https://img.shields.io/github/license/blackhc/llm-strategy)](https://img.shields.io/github/license/blackhc/llm-strategy)

Implementing the Strategy Pattern using LLMs.

Also, please see https://blog.blackhc.net/2022/12/llm_software_engineering/ for a wider perspective on why this could be important in the future.

This package adds a decorator `llm_strategy` that connects to an LLM (such as OpenAI’s GPT-3) and uses the LLM to "implement" abstract methods in interface classes. It does this by forwarding requests to the LLM and converting the responses back to Python data using Python's `@dataclasses`.

It uses the doc strings, type annotations, and method/function names as prompts for the LLM, and can automatically convert the results back into Python types (currently only supporting `@dataclasses`). It can also extract a data schema to send to the LLM for interpretation. While the `llm-strategy` package still relies on some Python code, it has the potential to reduce the need for this code in the future by using additional, cheaper LLMs to automate the parsing of structured data.

- **Github repository**: <https://github.com/blackhc/llm-strategy/>
- **Documentation** <https://blackhc.github.io/llm-strategy/>

## Research Example

The latest version also includes a package for hyperparameter tracking and collecting traces from LLMs.

This for example allows for meta optimization. See examples/research for a simple implementation using Generics.

You can find an example WandB trace at: https://wandb.ai/blackhc/blackboard-pagi/reports/Meta-Optimization-Example-Trace--Vmlldzo3MDMxODEz?accessToken=p9hubfskmq1z5yj1uz7wx1idh304diiernp7pjlrjrybpaozlwv3dnitjt7vni1j

The prompts showing off the pattern using Generics are straightforward:
```python
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
```

## Application Example

```python
from dataclasses import dataclass
from llm_strategy import llm_strategy
from langchain.llms import OpenAI


@llm_strategy(OpenAI(max_tokens=256))
@dataclass
class Customer:
    key: str
    first_name: str
    last_name: str
    birthdate: str
    address: str

    @property
    def age(self) -> int:
        """Return the current age of the customer.

        This is a computed property based on `birthdate` and the current year (2022).
        """

        raise NotImplementedError()


@dataclass
class CustomerDatabase:
    customers: list[Customer]

    def find_customer_key(self, query: str) -> list[str]:
        """Find the keys of the customers that match a natural language query best (sorted by closeness to the match).

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".

        Args:
            query: Natural language query

        Returns:
            The index of the best matching customer in the database.
        """
        raise NotImplementedError()

    def load(self):
        """Load the customer database from a file."""
        raise NotImplementedError()

    def store(self):
        """Store the customer database to a file."""
        raise NotImplementedError()


@llm_strategy(OpenAI(max_tokens=1024))
@dataclass
class MockCustomerDatabase(CustomerDatabase):
    def load(self):
        self.customers = self.create_mock_customers(10)

    def store(self):
        pass

    @staticmethod
    def create_mock_customers(num_customers: int = 1) -> list[Customer]:
        """
        Create mock customers with believable data (our customers are world citizens).
        """
        raise NotImplementedError()
```

See [examples/customer_database_search.py](examples/customer_database_search.py) for a full example.

![Customer Database Viewer](examples/app.svg)

![Searching for a Customer](examples/search1.svg)

![Searching for a Customer](examples/search2.svg)

## Getting started with contributing

Clone the repository first. Then, install the environment and the pre-commit hooks with 

```bash
make install
```

The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.

To finalize the set-up for publishing to PyPi or Artifactory, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting 
[this page](https://github.com/blackhc/llm-strategy/settings/secrets/actions/new).
- Create a [new release](https://github.com/blackhc/llm-strategy/releases/new) on Github. 
Create a new tag in the form ``*.*.*``.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
