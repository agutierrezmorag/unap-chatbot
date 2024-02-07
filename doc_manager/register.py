import streamlit as st
import streamlit_authenticator as stauth
from google.cloud import firestore
from google.oauth2 import service_account
import datetime
import re
import json

from utils import config


key_dict = json.loads(config.FIRESTORE_TEXT_KEY)
creds = service_account.Credentials.from_service_account_info(key_dict)
db = firestore.Client(credentials=creds)


def insert_user(email, username, password):
    """
    Inserta usuarios en la base de datos
    :param email:
    :param username:
    :param password:
    :return Usuario en caso de creación exitosa:
    """
    date_joined = datetime.datetime.now()

    # Referencia a la colección 'users'
    users_ref = db.collection('users')

    # Añade un documento con un ID autogenerado
    new_user_ref = users_ref.add({
        'email': email,
        'username': username,
        'password': password,
        'date_joined': date_joined
    })

    # Obtiene el ID del nuevo documento
    new_user_id = new_user_ref[1].id

    # Puedes devolver el nuevo usuario si es necesario
    new_user = {
        'id': new_user_id,
        'email': email,
        'username': username,
        'password': password,
        'date_joined': date_joined
    }

    return new_user


def fetch_users():
    """
    Recupera usuarios de la base de datos
    :return Lista de usuarios:
    """
    # Referencia a la colección 'users'
    users_ref = db.collection('users')

    # Obtiene todos los documentos en la colección
    users = users_ref.stream()

    # Lista para almacenar los usuarios
    user_list = []

    # Itera sobre los documentos y agrega los datos a la lista
    for user_doc in users:
        user_data = user_doc.to_dict()
        user_list.append(user_data)

    return user_list


def get_user_emails():
    """
    Obtiene los correos electrónicos de los usuarios
    :return Lista de correos electrónicos de usuarios:
    """
    # Referencia a la colección 'users'
    users_ref = db.collection('users')

    # Obtiene todos los documentos en la colección
    users = users_ref.stream()

    # Lista para almacenar los correos electrónicos de los usuarios
    emails = []

    # Itera sobre los documentos y agrega los correos electrónicos a la lista
    for user_doc in users:
        user_data = user_doc.to_dict()
        emails.append(user_data.get('email', ''))

    return emails


def get_usernames():
    """
    Obtiene los nombres de usuario de los usuarios
    :return Lista de nombres de usuario de usuarios:
    """
    # Referencia a la colección 'users'
    users_ref = db.collection('users')

    # Obtiene todos los documentos en la colección
    users = users_ref.stream()

    # Lista para almacenar los nombres de usuario de los usuarios
    usernames = []

    # Itera sobre los documentos y agrega los nombres de usuario a la lista
    for user_doc in users:
        user_data = user_doc.to_dict()
        usernames.append(user_data.get('username', ''))

    return usernames


def validate_email(email):
    """
    Verifica la validez del correo
    :param email:
    :return True si el email es valido si no retorna False:
    """
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$" #tesQQ12@gmail.com

    if re.match(pattern, email):
        return True
    return False


def validate_username(username):
    """
    Verifica la validez del nombre de usuario
    :param username:
    :return True si es valido si no retorna False:
    """

    pattern = "^[a-zA-Z0-9]*$"
    if re.match(pattern, username):
        return True
    return False


def sign_up():
    with st.form(key='signup', clear_on_submit=True):
        st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')

        if email:
            if validate_email(email):
                if email not in get_user_emails():
                    if validate_username(username):
                        if username not in get_usernames():
                            if len(username) >= 2:
                                if len(password1) >= 6:
                                    if password1 == password2:
                                        # Add User to DB
                                        hashed_password = stauth.Hasher([password2]).generate()
                                        insert_user(email, username, hashed_password[0])
                                        st.success('Cuenta creada exitosamente.')
                                        st.balloons()
                                    else:
                                        st.warning('Las constraseñas no coinciden.')
                                else:
                                    st.warning('Contraseña demasiado corta.')
                            else:
                                st.warning('Nombre de usuario demasiado corto.')
                        else:
                            st.warning('Nombre de usuario ya existente.')

                    else:
                        st.warning('Nombre de usuario no valido.')
                else:
                    st.warning('Correo ya en uso.')
            else:
                st.warning('Correo invalido')

        btn1, bt2, btn3, btn4, btn5 = st.columns(5)

        with btn3:
            st.form_submit_button('Registrarse')

# sign_uo()






