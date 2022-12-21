import typing

import docstring_parser.google
import docstring_parser.numpydoc
from docstring_parser import ParseError

GOOGLE_DEFAULT_SECTIONS = docstring_parser.google.DEFAULT_SECTIONS + [
    docstring_parser.google.Section("Prompt Template", "prompt_template", docstring_parser.google.SectionType.SINGULAR)
]
NUMPY_DEFAULT_SECTIONS = docstring_parser.numpydoc.DEFAULT_SECTIONS + [
    docstring_parser.numpydoc.Section("Prompt Template", "prompt_template")
]


def try_parse_docstring(
    text: str, parsers: typing.List[typing.Callable[[str], docstring_parser.Docstring]]
) -> docstring_parser.Docstring:
    exc: Exception | None = None
    rets = []
    for parser in parsers:
        try:
            ret = parser(text)
        except ParseError as ex:
            exc = ex
        else:
            rets.append(ret)

    if not rets:
        assert exc
        raise exc

    return sorted(rets, key=lambda d: len(d.meta), reverse=True)[0]


def extract_prompt_template(docstring: str) -> typing.Optional[str]:
    """Either return the full doc string, or if there is a `Prompt:` section, return only that section.

    Note that additional sections might appear before and after the `Prompt:` section.

    Examples:
        Examples using different doc strings styles.

        >>> example_google_docstring = '''
        ... This is a doc string.
        ...
        ... Prompt Template:
        ... This is the prompt.
        ...
        ... This is more doc string.
        ... '''
        >>> extract_prompt_template(example_google_docstring)
        'This is the prompt.'


        >>> example_numpy_docstring = '''
        ... This is a doc string.
        ...
        ... This is more doc string.
        ...
        ... Parameters
        ... ----------
        ... input : int
        ...     This is the input.
        ...
        ... Prompt Template
        ... ---------------
        ... This is the prompt.
        ...
        ... Returns
        ... -------
        ... output : int
        ...     This is the output.
        ...'''
        >>> extract_prompt_template(example_numpy_docstring)
        'This is the prompt.'
    """
    # TODO support ReST style doc strings
    # TODO support other doc string styles
    # Example:
    #
    #     >>> example_rest_docstring = '''
    #     ... This is a doc string.
    #     ...
    #     ... This is more doc string.
    #     ...
    #     ... :param input: This is the input.
    #     ... :type input: int
    #     ... :return: This is the output.
    #     ... :rtype: int
    #     ... :prompt_template: This is the prompt.
    #     ... '''
    #     >>> extract_prompt_template(example_rest_docstring)
    #     'This is the prompt.'
    #
    # This code is adapted from docstring_parser (MIT License):
    parsed_docstring = try_parse_docstring(
        docstring,
        [
            docstring_parser.google.GoogleParser(GOOGLE_DEFAULT_SECTIONS).parse,
            docstring_parser.numpydoc.NumpydocParser(NUMPY_DEFAULT_SECTIONS).parse,  # type: ignore
        ],
    )
    prompt_template = [section for section in parsed_docstring.meta if section.args == ["prompt_template"]]
    assert len(prompt_template) <= 1
    if prompt_template:
        return prompt_template[0].description
    return None
