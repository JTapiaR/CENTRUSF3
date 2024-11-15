# Instalar los paquetes necesarios
#!pip install streamlit requests crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29

import os
import json
import requests
import streamlit as st
from crewai import Agent, Task, Crew
from langchain.llms.base import LLM
from typing import Optional, List

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Configurar la clave API de Together AI

#TOGETHER_API_KEY= "7d6a9a16f62c1b6056b3a4e56b55c5528f5f86b0fca0a648553ac9c7e250e6cd" # Reemplaza con tu clave de Together AI
together_api_key = st.secrets["TOGETHER_API_KEY"]
# Función para interactuar con Llama 3.2 Vision
def llama32(messages, model_size=90):
    model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
    url = 'https://api.together.xyz/v1/completions'
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {together_api_key}"
    }
    res = json.loads(requests.post(url, headers=headers, data=json.dumps(payload)).content)

    if 'error' in res:
        raise Exception(res['error'])

    return res['choices'][0]['message']['content']

# Clase que envuelve la interacción con Llama 3.2 para adaptarla a CrewAI
class LlamaModel(LLM):
    model_size: int = 90
    stop: Optional[List[str]] = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is None:
            stop = self.stop
        messages = [{"role": "user", "content": prompt}]
        response = llama32(messages, self.model_size)
        if stop:
            for s in stop:
                response = response.split(s)[0]
        return response

    def bind(self, stop: Optional[List[str]] = None):
        return LlamaModel(model_size=self.model_size, stop=stop)

    @property
    def _llm_type(self) -> str:
        return "Llama32"

# Crear instancia del modelo Llama 3.2
llama_model = LlamaModel()

# Configuración de la app de Streamlit
st.title("AI agent specialized in climate risk ")
st.write("Genera un artículo que refleje las metricas de riesgo de un municipio basado en información pública disponbible, utilizando múltiples agentes para planificar, escribir y editar.")

# Input del usuario para el tema del artículo
topic = st.text_input("Introduce el tema para el artículo", "Describe el tipo de riesgo, y el municipio de tu interés")

# Definir Agentes utilizando el modelo LlamaModel
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="Estás planificando un artículo sobre el tema: {topic}. Recolectas información para ayudar a la audiencia a aprender y tomar decisiones informadas.",
    llm=llama_model,
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory="Estás escribiendo una nueva pieza de opinión basada en el esquema del planificador de contenido.",
    llm=llama_model,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="Revisas y editas para asegurar el cumplimiento de las mejores prácticas periodísticas.",
    llm=llama_model,
    allow_delegation=False,
    verbose=True
)

# Definir Tareas
plan_task = Task(
    description=(
        "1. Priorizar las fuentes de datos públicas disponibles actores clave y noticias relevantes sobre {topic}.\n"
        "2. Identificar la audiencia objetivo y desarrollar un esquema de contenido que incluya una introducción, puntos clave y un llamado a la acción.\n"
        "3. Incluir palabras clave de SEO y fuentes relevantes."
    ),
    expected_output="Un documento de plan de contenido detallado.",
    agent=planner,
)

write_task = Task(
    description=(
        "1. Usa el plan de contenido para crear una publicación de blog convincente sobre {topic}.\n"
        "2. Incorpora palabras clave de SEO de forma natural y proporciona un artículo bien estructurado."
    ),
    expected_output="Un artículo de blog bien escrito en formato markdown.",
    agent=writer,
)

edit_task = Task(
    description="Revisa errores gramaticales y alineación con la voz de la marca.",
    expected_output="Un artículo de blog bien editado en formato markdown.",
    agent=editor
)

# Crear la tripulación (Crew)
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=2
)

# Ejecutar la tripulación y generar el artículo
if st.button("Generar Artículo"):
    with st.spinner("Generando artículo..."):
        result = crew.kickoff(inputs={"topic": topic})
        
    st.markdown("### Artículo Generado:")
    st.markdown(result)

st.write("¡Proporciona un tema y haz clic en el botón para generar un artículo estructurado!")