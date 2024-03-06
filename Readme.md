# Proyecto de Sistemas de Recuperación de Información

## Autores

1. **Yoan Rene Ramos Corrales** - C412
2. **David Cabrera García** - C411

## Modelo Indexación de Semántica Latente (LSI)
El **Modelo de Indexación de Semántica Latente (LSI)** es una técnica de procesamiento de lenguaje natural que permite representar documentos de texto en un espacio semántico reducido. A través de la descomposición en valores singulares (SVD) de una matriz término-documento, LSI captura las relaciones semánticas entre palabras y documentos.

- **Matriz Término-Documento**: Representa una ponderación de términos en los documentos. En este caso usamos la ponderación tf-idf vista en conferencia.
- **Descomposición SVD**: Reduce la dimensionalidad de la matriz original y extrae los conceptos latentes.
- **Espacio Semántico Reducido**: Los documentos se representan en términos de estos conceptos latentes.

## Consideraciones al Desarrollar la Solución
Al desarrollar este proyecto, consideramos lo siguiente:

1. **Selección del Corpus**: Utilizamos el conjunto de datos **Cranfield**, que contiene documentos académicos para evaluar sistemas de recuperación de información.
2. **Preprocesamiento de Texto**: Realizamos limpieza de texto, eliminando stopwords, puntuación y normalizando términos.
3. **Implementación Eficiente**: Utilizamos técnicas de precomputación y optimización para acelerar el cálculo de similitudes entre consultas y documentos.

## Ejecución del Proyecto
Para ejecutar el proyecto, sigue estos pasos:

1. **Instalación de Dependencias**: Asegúrate de tener instaladas las bibliotecas necesarias (ir_datasets, numpy, sklearn, etc.).
2. **Carga del Conjunto de Datos**: Carga el conjunto de datos Cranfield utilizando `ir_datasets.load('cranfield')`.
3. **Precomputación**: Calcula las representaciones vectoriales precomputadas utilizando la clase `precomputer.Precomputed`.
4. **Entrenamiento del Modelo LSI**: Utiliza la clase `lsi_model.Lsi_model` para entrenar el modelo LSI.
5. **Consulta**: Define una consulta utilizando la clase `query.Query` y obtén los documentos más similares.

## Solución Desarrollada
Nuestra solución consiste en:

1. **Representación Vectorial**: Transformamos los documentos en vectores utilizando LSI.
2. **Cálculo de Similitud**: Utilizamos la similitud del coseno para encontrar los documentos más relevantes para una consulta dada.

### Definiciones Matemáticas
- **Matriz Término-Documento**: $X$, donde $X_{ij}$ representa el **TF-IDF** $i$ en el documento $j$.
- **Descomposición SVD**: $X = U \Sigma V^T$, donde $U$ y $V$ son matrices ortogonales y $\Sigma$ es una matriz diagonal con los valores singulares.
- **Representación LSI**: $X_{\text{LSI}} = U_k \Sigma_k V_k^T$, donde $k$ es la dimensión reducida.

#### Fórmulas adicionales:
1. **TF (Frecuencia del Término)**:
   - $\text{TF}(t,d) = \frac{f(t,d)}{\sum_{t' \in d} f(t',d)}$, donde $f(t,d)$ es el recuento crudo del término $t$ en el documento $d$.

2. **IDF (Inverso de la Frecuencia del Documento)**:
   - $\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$, donde $N$ es el número total de documentos y $\text{df}(t)$ es la frecuencia de documentos que contienen el término $t$.

3. **TF-IDF (Frecuencia del Término-Inverso de la Frecuencia del Documento)**:
   - $\text{TF-IDF}(t,d) = \text{TF}(t,d) \cdot \text{IDF}(t)$

## Insuficiencias y Mejoras Propuestas
Algunas insuficiencias de nuestra solución son:

1. **Sensibilidad a Términos Raros**: LSI puede no manejar bien términos raros o poco frecuentes.
2. **Dimensionalidad Reducida**: La reducción de dimensionalidad puede perder información relevante.

Para mejorar la solución, podríamos considerar:

1. **Modelos Más Avanzados**: Explorar modelos más avanzados como LDA o word embeddings.
2. **Ajuste de Parámetros**: Experimentar con diferentes dimensiones y parámetros de LSI.