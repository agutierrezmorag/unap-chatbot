import time
import uuid

import streamlit as st
import streamlit_authenticator as stauth
from st_pages import show_pages_from_config

from doc_manager.github_management import (
    add_files_to_repo,
    delete_repo_doc,
    get_repo_docs_as_pd,
)
from doc_manager.pinecone_management import (
    delete_namespace,
    get_index_data,
    process_and_load_documents,
)
from doc_manager.register import fetch_users

logo_path = "logos/unap_negativo.png"


def reset_state_and_rerun(state_key):
    st.session_state[state_key] = False
    st.rerun()


def update_session_and_rerun(upload_key):
    st.session_state[upload_key] = str(uuid.uuid4())
    st.rerun()


def general_info():
    st.header("📚 Administración de documentos")
    st.markdown(
        "Esta herramienta permite administrar los documentos en memoria. \
        Es posible ver, subir y eliminar documentos del repositorio. \
        Todo documento subido al repositorio se procesa automáticamente y se almacena en memoria para ser consultado por la IA."
    )

    st.warning(
        "Para evitar errores, no recargue ni cierre la página mientras se esté realizando una operación.",
        icon="⚠️",
    )
    index_data = get_index_data()

    space_used = index_data.index_fullness
    percentage = 100 - (space_used * 100)
    st.progress(
        1 - space_used,
        f"{percentage:.2f}% de espacio disponible en memoria",
    )


def manage_docs(
    doc_type: str,
    upload_key: str,
    delete_doc_key: str,
    namespace: str,
):
    form = st.form(key=f"{namespace}_list_form", border=False)

    df = get_repo_docs_as_pd(namespace)
    with form:
        st.data_editor(
            df,
            key=f"{namespace}_list_df",
            hide_index=True,
            height=300,
            use_container_width=True,
            column_order=["selected", "name", "html_url", "download_url", "size"],
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "Seleccionar",
                    width="small",
                ),
                "name": st.column_config.TextColumn(
                    "📄 Nombre",
                    width="medium",
                ),
                "html_url": st.column_config.LinkColumn(
                    "🔗 URL",
                    display_text="Ver en GitHub",
                    width="small",
                ),
                "download_url": st.column_config.LinkColumn(
                    "⬇️ Descarga",
                    display_text="Descargar",
                    width="small",
                ),
                "size": st.column_config.NumberColumn(
                    "📏 Tamaño (Kb)",
                    format="%.1f",
                    width="small",
                ),
            },
            disabled=["name", "html_url", "download_url", "size"],
        )
    selected_rows = st.session_state[f"{namespace}_list_df"]["edited_rows"]

    delete_confirmation_dialog = st.empty()
    delete_action_button = st.empty()

    delete_action_button = form.form_submit_button(
        f"Eliminar {doc_type}s seleccionados",
        use_container_width=True,
        disabled=df.empty,
    )

    if st.session_state.get(delete_doc_key):
        delete_confirmation_dialog.warning(
            "¿Seguro que desea eliminar los documentos seleccionados?",
            icon="⚠️",
        )
        if st.button(
            f"Eliminar {namespace} seleccionados",
            use_container_width=True,
            type="primary",
        ):
            selected_indices = list(
                st.session_state[f"{namespace}_list_df"]["edited_rows"].keys()
            )
            selected_file_paths = df.loc[selected_indices, "path"].tolist()
            delete_repo_doc(
                file_paths=selected_file_paths,
                namespace=namespace,
            )
            time.sleep(2)
            reset_state_and_rerun(delete_doc_key)
        elif st.button("Cancelar", use_container_width=True, key=str(uuid.uuid4())):
            reset_state_and_rerun(delete_doc_key)
    elif delete_action_button:
        if selected_rows:
            st.session_state[delete_doc_key] = True
            st.rerun()
        else:
            delete_confirmation_dialog.error(
                "No se ha seleccionado ningún documento para eliminar.",
                icon="❌",
            )

    uploaded_files = st.file_uploader(
        f"Subir archivo .{doc_type}",
        type=doc_type,
        accept_multiple_files=True,
        help=f"Selecciona uno o más archivos. Solo se permiten {doc_type}.",
        key=st.session_state[upload_key],
    )

    if uploaded_files:
        if st.button(
            f"Subir archivos de {namespace} al repositorio",
            use_container_width=True,
            type="primary",
        ):
            add_files_to_repo(
                file_list=uploaded_files,
                namespace=namespace,
            )
            update_session_and_rerun(upload_key)

        if st.button(
            "Limpiar",
            use_container_width=True,
            key=str(uuid.uuid4()),
        ):
            update_session_and_rerun(upload_key)


def on_delete_button_click(selected_file_paths, namespace, delete_doc_key):
    delete_repo_doc(
        file_paths=selected_file_paths,
        namespace=namespace,
    )
    time.sleep(2)
    reset_state_and_rerun(delete_doc_key)


def wikipedia():
    index_data = get_index_data()
    knows_wikipedia = "Wikipedia" in index_data.namespaces

    if knows_wikipedia:
        st.success(
            "Actualmente, la IA SI conoce el contenido de la página de Wikipedia.",
            icon="✅",
        )
    else:
        st.warning(
            "Actualmente, la IA NO conoce el contenido de la página de Wikipedia.",
            icon="⚠️",
        )
    st.markdown(
        "El contenido de la página de Wikipedia de la [Universidad Arturo Prat](https://es.wikipedia.org/wiki/Universidad_Arturo_Prat) "
        "está disponible para añadirse a la base de conocimientos de la IA. "
        "Este contenido puede ser útil para responder preguntas generales sobre la universidad o sobre datos que no estén en los reglamentos. "
        "Si se añade, la IA podrá responder preguntas basándose en esta información."
    )
    st.markdown(
        "Cada vez que se realice esta operación, el contenido anterior de la página de Wikipedia es olvidado y se reemplazará automáticamente "
        "por el contenido nuevo. Se recomienda hacerlo solo si se está seguro de que el contenido es relevante y actualizado."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Añadir contenido de Wikipedia",
            use_container_width=True,
            type="primary",
        ):
            process_and_load_documents(namespace="Wikipedia")
            st.toast("Contenido de Wikipedia añadido a memoria.", icon="✅")
    with col2:
        if st.button(
            "Eliminar contenido de Wikipedia",
            use_container_width=True,
            type="secondary",
        ):
            delete_namespace("Wikipedia")
            st.toast("Contenido de Wikipedia eliminado de memoria", icon="⚠️")


def main():
    st.set_page_config(
        page_title="Chatbot UNAP",
        page_icon="🤖",
    )

    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: auto;
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(logo_path, use_column_width=True)
    show_pages_from_config()

    if "txt_upload_key" not in st.session_state:
        st.session_state.txt_upload_key = str(uuid.uuid4())
    if "txt_delete_state" not in st.session_state:
        st.session_state.txt_delete_state = False

    if "calendar_upload_key" not in st.session_state:
        st.session_state.calendar_upload_key = str(uuid.uuid4())
    if "calendar_delete_state" not in st.session_state:
        st.session_state.calendar_delete_state = False

    if "web_upload_key" not in st.session_state:
        st.session_state.web_upload_key = str(uuid.uuid4())
    if "web_delete_state" not in st.session_state:
        st.session_state.web_delete_state = False

    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user["email"])
            usernames.append(user["username"].lower())
            passwords.append(user["password"])

        credentials = {"usernames": {}}
        for index in range(len(emails)):
            credentials["usernames"][usernames[index]] = {
                "name": emails[index],
                "password": passwords[index],
            }

        Authenticator = stauth.Authenticate(
            credentials, cookie_name="Streamlit", key="abcdef", cookie_expiry_days=1
        )

        email, authentication_status, username = Authenticator.login(
            "main",
            fields={
                "Form name": "Inicio de sesión",
                "Username": "Usuario",
                "Password": "Contraseña",
                "Login": "Iniciar Sesión",
            },
        )

        # Para querer registrar una cuenta.
        # if not authentication_status:
        # sign_up()

        if not username:
            st.warning("Por favor, ingrese sus credenciales.")
            st.stop()
        elif username not in usernames:
            st.warning("Usuario no existente.")
            st.stop()
        if not authentication_status:
            st.error("Contraseña o usuario incorrectos.")
            st.stop()
        else:
            st.sidebar.subheader(f"Bienvenido {username}")
            Authenticator.logout("Cerrar Sesión", "sidebar")

            general_info()

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📃 Reglamentos", "🗓️ Calendarios", "🔗 Web", "🌍 Wikipedia"]
            )

            with tab1:
                manage_docs(
                    "txt",
                    "txt_upload_key",
                    "txt_delete_state",
                    "Reglamentos",
                )

            with tab2:
                manage_docs(
                    "xml",
                    "calendar_upload_key",
                    "calendar_delete_state,",
                    "Calendarios",
                )

            with tab3:
                manage_docs(
                    "xml",
                    "web_upload_key",
                    "web_delete_state,",
                    "Web",
                )

            with tab4:
                wikipedia()

    except Exception as e:
        print(e)
        st.rerun()


if __name__ == "__main__":
    main()
