from llm_strategy.prompt_doc_string import extract_prompt_template


def test_extract_prompt_template():
    example_google_docstring = """
    This is a doc string.

    Prompt Template:
        This is the prompt.
    """
    assert extract_prompt_template(example_google_docstring) == "This is the prompt."

    example_numpy_docstring = """
    This is a doc string.

    This is more doc string.

    Parameters
    ----------
    input : int
        This is the input.

    Prompt Template
    ---------------
    This is the prompt.

    Returns
    -------
    output : int
        This is the output.
    """
    assert extract_prompt_template(example_numpy_docstring) == "This is the prompt."

    example_no_prompt_docstring = """
    This is a doc string.

    This is more doc string.
    """
    assert extract_prompt_template(example_no_prompt_docstring) is None

    example_no_prompt_docstring = """
    This is a doc string.

    This is more doc string.

    Parameters
    ----------
    input : int
        This is the input.

    Returns
    -------
    output : int
        This is the output.
    """
    assert extract_prompt_template(example_no_prompt_docstring) is None
