from langchain_community.callbacks.streamlit.streamlit_callback_handler import (
    LLMThoughtLabeler,
)


class CustomLLMThoughtLabeler(LLMThoughtLabeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_initial_label(self) -> str:
        """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
        return "🤔 **Pensando...**"

    def get_history_label(self) -> str:
        """Return a markdown label for the special 'history' container
        that contains overflow thoughts.
        """
        return "📜 **Historial**"

    def get_final_agent_thought_label(self) -> str:
        """Return the markdown label for the agent's final thought -
        the "Now I have the answer" thought, that doesn't involve
        a tool.
        """
        return "🤗 **Respuesta generada**"
