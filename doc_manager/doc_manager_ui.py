import time
import uuid

import streamlit as st
import streamlit_authenticator as stauth
from st_pages import show_pages_from_config

from doc_manager.github_management import (
    add_files_to_repo,
    delete_repo_doc,
    get_repo_docs_as_pd,
    get_repo_documents,
    upload_repo_docs,
)
from doc_manager.pinecone_management import (
    delete_all_namespaces,
    delete_namespace,
    get_index_data,
)
from doc_manager.register import fetch_users
from doc_manager.wikipedia_management import upload_wikipedia_page

logo_path = "logos/unap_negativo.png"


def _general_info_section():
    st.markdown(
        "En esta sección se pueden gestionar los documentos del repositorio. "
        "Es posible ver los documentos presentes en el repositorio, "
        "subir nuevos documentos o eliminar documentos ya existentes."
    )

    index_data = get_index_data()

    space_used = index_data.index_fullness
    st.progress(
        1 - space_used,
        f"{100-space_used:.2f}% espacio disponible en memoria de la IA",
    )


def _doc_list_section():
    progress_bar_placeholder = st.empty()
    container_placeholder = st.empty()
    form = st.form(key="document_list_form", border=False)

    df = get_repo_docs_as_pd("txt")
    if df.empty:
        container_placeholder.warning("No hay documentos en el repositorio.", icon="⚠️")
    else:
        with form:
            st.data_editor(
                df,
                key="document_list_df",
                hide_index=True,
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
        selected_rows = st.session_state.document_list_df["edited_rows"]

    confirm_dialog = st.empty()
    action_button = st.empty()

    action_button = form.form_submit_button(
        "Eliminar documentos seleccionados",
        use_container_width=True,
        disabled=df.empty,
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        confirm_button = st.empty()
    with bcol2:
        cancel_button = st.empty()

    if st.session_state.get("delete_selected_docs"):
        confirm_dialog.warning(
            "¿Seguro que desea eliminar los documentos seleccionados?",
            icon="⚠️",
        )
        if confirm_button.button("Confirmar", use_container_width=True, type="primary"):
            for index in selected_rows:
                doc_name = df.loc[index, "name"]
                doc_path = df.loc[index, "path"]
                if delete_repo_doc(doc_path):
                    st.toast(
                        f"Documento '{doc_name}' eliminado.",
                        icon="⚠️",
                    )
            get_repo_documents.clear()
            st.session_state.delete_selected_docs = False
            time.sleep(1)
            st.rerun()
        elif cancel_button.button("Cancelar", use_container_width=True):
            st.session_state.delete_selected_docs = False
            st.rerun()
    elif action_button:
        if selected_rows:
            st.session_state.delete_selected_docs = True
            st.rerun()
        else:
            confirm_dialog.error(
                "No se ha seleccionado ningún documento para eliminar.",
                icon="❌",
            )

    uploaded_files = st.file_uploader(
        "Sube un nuevo documento",
        type="txt",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos de texto. Solo se permiten archivos .txt.",
        key=st.session_state.upload_key,
    )

    if uploaded_files:
        if st.button("Subir archivos"):
            if uploaded_files:
                add_files_to_repo(
                    uploaded_files,
                    "txt",
                    container_placeholder,
                    progress_bar_placeholder,
                )
                st.session_state.upload_key = str(uuid.uuid4())
                get_repo_documents.clear()
                st.rerun()

        if st.button("Limpiar"):
            st.session_state.upload_key = str(uuid.uuid4())
            st.rerun()

    st.info(
        "**Importante**: La IA solo sera consciente de que ha habido modificaciones "
        "en los documentos una vez se [registren los cambios](#Registro).",
        icon="📢",
    )


def _calendar_list_section():
    progress_bar_placeholder = st.empty()
    container_placeholder = st.empty()
    form = st.form(key="calendar_list_form", border=False)

    df = get_repo_docs_as_pd("pdf")
    if df.empty:
        container_placeholder.warning("No hay documentos en el repositorio.", icon="⚠️")
    else:
        with form:
            st.data_editor(
                df,
                key="calendar_list_df",
                hide_index=True,
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
        selected_rows = st.session_state.calendar_list_df["edited_rows"]

    confirm_dialog = st.empty()
    action_button = st.empty()

    action_button = form.form_submit_button(
        "Eliminar calendarios seleccionados",
        use_container_width=True,
        disabled=df.empty,
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        confirm_button = st.empty()
    with bcol2:
        cancel_button = st.empty()

    if st.session_state.get("delete_selected_calendars"):
        confirm_dialog.warning(
            "¿Seguro que desea eliminar los documentos seleccionados?",
            icon="⚠️",
        )
        if confirm_button.button(
            "Confirmarr", use_container_width=True, type="primary"
        ):
            for index in selected_rows:
                doc_name = df.loc[index, "name"]
                doc_path = df.loc[index, "path"]
                if delete_repo_doc(doc_path):
                    st.toast(
                        f"Documento '{doc_name}' eliminado.",
                        icon="⚠️",
                    )
            get_repo_documents.clear()
            st.session_state.delete_selected_calendars = False
            time.sleep(1)
            st.rerun()
        elif cancel_button.button("Cancelarr", use_container_width=True):
            st.session_state.delete_selected_calendars = False
            st.rerun()
    elif action_button:
        if selected_rows:
            st.session_state.delete_selected_calendars = True
            st.rerun()
        else:
            confirm_dialog.error(
                "No se ha seleccionado ningún documento para eliminar.",
                icon="❌",
            )

    uploaded_files = st.file_uploader(
        "Sube un nuevo calendario",
        type="pdf",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos. Solo se permiten PDF.",
        key=st.session_state.calendar_upload_key,
    )

    if uploaded_files:
        if st.button("Subir calendarios"):
            if uploaded_files:
                add_files_to_repo(
                    uploaded_files,
                    "pdf",
                    container_placeholder,
                    progress_bar_placeholder,
                )
                st.session_state.calendar_upload_key = str(uuid.uuid4())
                get_repo_documents.clear()
                st.rerun()

        if st.button("Limpiar"):
            st.session_state.calendar_upload_key = str(uuid.uuid4())
            st.rerun()

    st.info(
        "**Importante**: La IA solo sera consciente de que ha habido modificaciones "
        "en los documentos una vez se [registren los cambios](#Registro).",
        icon="📢",
    )


def _wikipedia_section():
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
    st.caption(
        "No es necesario realizar el proceso de registro de cambios para que la IA conozca el contenido de Wikipedia, esto se hace automáticamente."
    )
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

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Añadir contenido de Wikipedia",
            use_container_width=True,
            type="primary",
        ):
            upload_wikipedia_page()
            time.sleep(10)
            st.rerun()
    with col2:
        if st.button(
            "Eliminar contenido de Wikipedia",
            use_container_width=True,
            type="secondary",
        ):
            delete_namespace("Wikipedia")
            time.sleep(4)
            st.rerun()


def save_changes_section():
    st.markdown(
        "Cuando se presione el botón `Registrar cambios`, los documentos que se hayan subido se procesan y \
                se integran en la base de conocimientos de la IA. Solo a partir de ese momento, la IA podrá responder \
                preguntas basándose en la información contenida en estos documentos."
    )
    st.info(
        "Este proceso puede tardar varios minutos. No refresque la página mientras se esté realizando el registro. \
        Es posible que la IA no responda preguntas mientras se esté realizando el registro.",
        icon="💡",
    )

    confirm_dialog = st.empty()
    save_changes_button = st.empty()
    delete_mem_button = st.empty()
    cancel_button = st.empty()

    if st.session_state.get("delete_all_mem"):
        confirm_dialog.error(
            "Esto eliminará **TODA** la memoria de la IA. ¿Está seguro de que desea continuar?",
            icon="❌",
        )
        if delete_mem_button.button("Confirmar"):
            delete_all_namespaces()
            st.session_state.delete_all_mem = False
            time.sleep(2)
            st.rerun()
        elif cancel_button.button("Cancelar"):
            st.session_state.delete_all_mem = False
            st.rerun()
    else:
        if save_changes_button.button(
            "Registrar cambios", use_container_width=True, type="primary"
        ):
            upload_repo_docs()
            st.success("Cambios registrados exitosamente.", icon="✅")
            time.sleep(10)
            st.rerun()

        if delete_mem_button.button(
            "Eliminar memoria de la IA",
            use_container_width=True,
        ):
            st.session_state.delete_all_mem = True
            st.rerun()


def main():
    st.set_page_config(
        page_title="Administrador de Documentos UNAP",
        page_icon="📚",
        initial_sidebar_state="collapsed",
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

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = str(uuid.uuid4())
    if "calendar_upload_key" not in st.session_state:
        st.session_state.calendar_upload_key = str(uuid.uuid4())
    if "delete_selected" not in st.session_state:
        st.session_state.delete_selected = False

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

            st.header("📚 Gestión de documentos", divider=True)
            _general_info_section()

            st.header("🗃️ Documentos en el repositorio", divider=True)
            st.markdown(
                "Listado de documentos presentes en el repositorio. Es posible seleccionar uno o más documentos para eliminarlos."
            )
            _doc_list_section()

            st.header("🗃️ Calendarios", divider=True)
            _calendar_list_section()

            st.header("🌐 Wikipedia", divider=True)
            _wikipedia_section()

            st.header("💾 Registrar cambios", anchor="Registro", divider="red")
            save_changes_section()

    except Exception as e:
        print(e)
        st.rerun()


if __name__ == "__main__":
    main()
