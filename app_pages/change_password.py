import streamlit as st
import yaml

if __name__ == "__page__":
    try:
        if st.session_state.authenticator.reset_password(
            st.session_state["username"],
            key=st.session_state.reset_password_key,
            fields={
                "Form name": "Cambiar contraseña",
                "Current password": "Contraseña actual",
                "New password": "Contraseña nueva",
                "Repeat password": "Repetir contraseña",
                "Reset": "Actualizar",
            },
        ):
            with open(".streamlit/auth_config.yaml", "w") as file:
                yaml.dump(st.session_state.auth_config, file, default_flow_style=False)
            st.success("Se ha actualizado la contraseña. Vuelva a iniciar sesión.")
            st.session_state.role = "User"
    except Exception as e:
        st.error(f"Error al actualizar la contraseña: {e}")
