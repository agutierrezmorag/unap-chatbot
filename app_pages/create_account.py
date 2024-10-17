import streamlit as st
import yaml

if __name__ == "__page__":
    try:
        (
            email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user,
        ) = st.session_state.authenticator.register_user()
        if email_of_registered_user:
            with open(".streamlit/auth_config.yaml", "w") as file:
                yaml.dump(st.session_state.auth_config, file, default_flow_style=False)
            st.success("Se ha creado la cuenta correctamente.")
    except Exception as e:
        st.error(e)
