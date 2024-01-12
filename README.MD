# UNAP Chatbot 🤖

Desarrollo de chatbot consciente de documentos y reglamentos pertinentes a la universidad Arturo Prat y la carrera de Ingenieria Civil en Computacion e Informatica.

Disponible en: <https://unap-chatbot.streamlit.app/>

## Servicios utilizados

- [**OpenAI API**](https://openai.com/blog/openai-api) para la generacion de las respuestas. Modelo `gpt-3.5-turbo`.
- [**Pinecone**](https://www.pinecone.io/) como base de datos de vectores.
- [**Streamlit**](https://streamlit.io/) para el frontend y deploy de la herramienta.

## To-Do

- [ ] Implementar **base de datos de vectores** local y verificar si existe diferencia significativa con Pinecone.
- [ ] Reemplazar los **embeddings** de OpenAI *(OpenAIEmbeddings)* con embeddings gratuitos y comprobar si existe alguna diferencia significativa.
- [ ] Realizar **pruebas de rendimiento** del chat, para tener un aproximado de cuantos usuarios simultaneos permite la pag. web.
- [ ] Implementar **autentificacion** en la pagina `📑 Documentos` [conversar implementacion con el profesor].
- [ ] Implementar boton para realizar **vectorizado** de los documentos subidos en la pagina `📑 Documentos`.
- [ ] Realizar **pruebas con modelos de chat gratuitos** (HuggingFace) y comparar respuestas con las obtenidas con los modelos de OpenAI.
- [ ] Implementar el **streaming de respuestas** *(mostrar la respuesta en pantalla a medida que se va generando)*.
- [ ] Discutir el diseño **frontend** con el profesor y realizar cambios donde sea necesario.
  - [ ] Presentar la pagina como propiedad de la UNAP. Implementar sus colores o logo en alguna parte.
  - [ ] Presentar los documentos que conoce la IA de manera mejor.

Taller de titulo. UNAP 2023.
