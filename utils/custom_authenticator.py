from streamlit_authenticator.views.authentication_view import Authenticate

TRANSLATED_PASSWORD_INSTRUCTIONS = """
**La contraseña debe cumplir con los siguientes requisitos:**
- Entre 8 y 20 caracteres.
- Al menos una letra minúscula.
- Al menos una letra mayúscula.
- Al menos un número.
- Al menos un caracter especial de [@$!%*?&].
"""


class CustomAuthenticate(Authenticate):
    def __init__(self, *args, **kwargs):
        # Override the PASSWORD_INSTRUCTIONS attribute
        self.PASSWORD_INSTRUCTIONS = TRANSLATED_PASSWORD_INSTRUCTIONS
        super().__init__(*args, **kwargs)

    def register_user(self, *args, **kwargs):
        # Override the password instructions in the register_user method
        self.attrs["password_instructions"] = self.PASSWORD_INSTRUCTIONS
        return super().register_user(*args, **kwargs)

    def login(self, *args, **kwargs):
        # Override the password instructions in the login method
        self.attrs["password_instructions"] = self.PASSWORD_INSTRUCTIONS
        return super().login(*args, **kwargs)

    def reset_password(self, *args, **kwargs):
        # Override the password instructions in the reset_password method
        self.attrs["password_instructions"] = self.PASSWORD_INSTRUCTIONS
        return super().reset_password(*args, **kwargs)
