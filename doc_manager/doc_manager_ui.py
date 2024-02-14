import datetime
import uuid

import streamlit as st
import streamlit_authenticator as stauth
from st_pages import show_pages_from_config

from doc_manager.github_management import (
    add_files_to_repo,
    delete_repo_doc,
    get_repo_docs_as_pd,
    get_repo_documents,
)
from doc_manager.pinecone_management import (
    delete_namespace,
    get_index_data,
    process_and_load_documents,
)
from doc_manager.register import fetch_users

logo_path = "logos/unap_negativo.png"


def general_info():
    st.markdown(
        """
        # 📑 Información

        ## 💬 Gestión de Documentos

        1. **Visualización de Documentos**: En la interfaz principal, encontrará todos los documentos que han sido cargados al repositorio.

        2. **Carga de Documentos**: Para incorporar un nuevo documento al repositorio, presione 'Cargar', seleccione el documento deseado y este se subirá de forma automática.

        3. **Eliminación de Documentos**: Si desea eliminar un documento del repositorio, selecciónelo y presione 'Eliminar'. El documento será suprimido inmediatamente del repositorio.

        ## 🧠 Memoria de la IA

        Es importante resaltar que, aunque un documento se encuentre en el repositorio, la IA no estará consciente de su contenido hasta que se actualice su memoria. Para ello, es necesario presionar 'Guardar Cambios' tras la carga o eliminación de un documento.

        Por favor, recuerde que cada vez que añada o suprima uno o más documentos, debe presionar 'Guardar Cambios' para asegurar que la IA esté al corriente de los cambios.

        Con estos pasos, podrá gestionar documentos de manera eficiente y segura, y mantener actualizada la memoria de la IA.
        """
    )

    index_data = get_index_data()

    space_used = index_data.index_fullness
    percentage = 100 - (space_used * 100)
    st.progress(
        1 - space_used,
        f"{percentage:.2f}% espacio disponible en memoria",
    )


def manage_docs(
    doc_type: str,
    upload_key: str,
    delete_doc_key: str,
    delete_mem_key: str,
    register_button_text: str,
    register_type: str,
):
    if register_type == "Reglamentos":
        last_update = get_last_doc_update()
    elif register_type == "Calendarios":
        last_update = get_last_calendar_update()
    st.info(f"La memoria fue actualizada por última vez el: {last_update}", icon="ℹ️")

    progress_bar_placeholder = st.empty()
    container_placeholder = st.empty()
    form = st.form(key=f"{doc_type}_list_form", border=False)

    df = get_repo_docs_as_pd(doc_type)
    if df.empty:
        container_placeholder.warning("No hay documentos en el repositorio.", icon="⚠️")
    else:
        with form:
            st.data_editor(
                df,
                key=f"{doc_type}_list_df",
                hide_index=True,
                height=150,
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
        selected_rows = st.session_state[f"{doc_type}_list_df"]["edited_rows"]

    confirm_dialog = st.empty()
    action_button = st.empty()

    action_button = form.form_submit_button(
        f"Eliminar {doc_type}s seleccionados",
        use_container_width=True,
        disabled=df.empty,
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        confirm_button = st.empty()
    with bcol2:
        cancel_button = st.empty()

    if st.session_state.get(delete_doc_key):
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
            st.session_state[delete_doc_key] = False
            st.rerun()
        elif cancel_button.button("Cancelar", use_container_width=True):
            st.session_state[delete_doc_key] = False
            st.rerun()
    elif action_button:
        if selected_rows:
            st.session_state[delete_doc_key] = True
            st.rerun()
        else:
            confirm_dialog.error(
                "No se ha seleccionado ningún documento para eliminar.",
                icon="❌",
            )

    uploaded_files = st.file_uploader(
        f"Subir {doc_type}",
        type=doc_type,
        accept_multiple_files=True,
        help=f"Selecciona uno o más archivos. Solo se permiten {doc_type}.",
        key=st.session_state[upload_key],
    )

    if uploaded_files:
        if st.button(f"Subir {doc_type}s"):
            if uploaded_files:
                add_files_to_repo(
                    uploaded_files,
                    doc_type,
                    container_placeholder,
                    progress_bar_placeholder,
                )
                st.session_state[upload_key] = str(uuid.uuid4())
                get_repo_documents.clear()
                st.rerun()

        if st.button("Limpiar"):
            st.session_state[upload_key] = str(uuid.uuid4())
            st.rerun()

    confirm_dialog = st.empty()
    save_changes_button = st.empty()
    delete_mem_button = st.empty()
    cancel_button = st.empty()

    if st.session_state.get(delete_mem_key):
        confirm_dialog.error(
            f"Esto eliminará **TODA** la memoria de la IA sobre {register_type}. ¿Está seguro de que desea continuar?",
            icon="🚩",
        )
        if delete_mem_button.button("Confirmar", key=f"confirm_delete_{doc_type}"):
            delete_namespace(register_type)
            st.toast("Memoria eliminada.", icon="⚠️")
            st.session_state[delete_mem_key] = False
            st.rerun()
        elif cancel_button.button("Cancelar", key=f"cancel_delete_{doc_type}"):
            st.session_state[delete_mem_key] = False
            st.rerun()
    else:
        if save_changes_button.button(
            register_button_text,
            use_container_width=True,
            type="primary",
            disabled=df.empty,
        ):
            process_and_load_documents(namespace=register_type, directory_path=doc_type)
            if register_type == "Reglamentos":
                get_last_doc_update.clear()
                st.session_state.last_doc_update = datetime.datetime.now().strftime(
                    "%d/%m/%Y %H:%M hrs."
                )
            elif register_type == "Calendarios":
                get_last_calendar_update.clear()
                st.session_state.last_calendar_update = (
                    datetime.datetime.now().strftime("%d/%m/%Y %H:%M hrs.")
                )
            st.success(f"{register_type} registrados exitosamente.", icon="✅")
            st.rerun()

        if delete_mem_button.button(
            f"Eliminar memoria de la IA sobre {register_type}",
            use_container_width=True,
        ):
            st.session_state[delete_mem_key] = True
            st.rerun()


def wikipedia():
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


@st.cache_data(show_spinner=False)
def get_last_doc_update():
    if "last_doc_update" in st.session_state:
        return st.session_state["last_doc_update"]
    return "Nunca"


@st.cache_data(show_spinner=False)
def get_last_calendar_update():
    if "last_calendar_update" in st.session_state:
        return st.session_state["last_calendar_update"]
    return "Nunca"


def main():
    st.set_page_config(
        page_title="Administración de Documentos UNAP",
        page_icon=":gear:",
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
    if "delete_txt_key" not in st.session_state:
        st.session_state.delete_txt_key = False
    if "delete_pdf_key" not in st.session_state:
        st.session_state.delete_pdf_key = False

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

            tab1, tab2, tab3 = st.tabs(
                ["📃 Reglamentos", "🗓️ Calendarios", "🌍 Wikipedia"]
            )

            with tab1:
                manage_docs(
                    "txt",
                    "upload_key",
                    "delete_txt_key",
                    "delete_selected_docs",
                    "Registrar reglamentos",
                    "Reglamentos",
                )

            with tab2:
                manage_docs(
                    "pdf",
                    "calendar_upload_key",
                    "delete_pdf_key,",
                    "delete_selected_calendars",
                    "Registrar calendarios",
                    "Calendarios",
                )

            with tab3:
                wikipedia()

    except Exception as e:
        print(e)
        st.rerun()


if __name__ == "__main__":
    main()
