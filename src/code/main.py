import ir_datasets
import numpy as np
import query
import metrics
import lsi_model 
from sklearn.metrics.pairwise import cosine_similarity
import precomputer
from preprocess import preprocess
from boolean_model import BooleanModel
import tkinter as tk
from tkinter import ttk

def evaluate_models():

    doc_ids=[doc.doc_id for doc in dataset.docs_iter()]
    dic={}
    for qrel in dataset.qrels_iter():
        if qrel.relevance>=3:
            if qrel.query_id not in dic:
                dic[qrel.query_id]=list()
            dic[qrel.query_id].append(qrel.doc_id)



    lsi_recovered={}
    boolean_recovered={}
    for query_ in dataset.queries_iter():
        #print(query_.query_id)
        if query_.query_id not in dic: 
            continue
        
        lsi_recovered[query_.query_id]=lsi.get_filtered_docs(query_.text,0.6)
        boolean_recovered[query_.query_id] = boolean.get_matching_docs(query_.text)


    for i in lsi_recovered.keys():
        recuperado=lsi_recovered[i]
        relevante=dic[i]
        relevante_recuperado=set(recuperado).intersection(set(relevante))
        relevante_no_recuperado=set(relevante).difference(set(recuperado))
        irrelevante_recuperado=set(recuperado).difference(set(relevante))
        irrelevante_no_recuperado=(set(doc_ids).difference(set(recuperado))).difference(set(relevante))
        print('   ')
        print(len(relevante_recuperado))
        print(len(relevante_no_recuperado))
        print(len(irrelevante_recuperado))
        print(len(irrelevante_no_recuperado))
        metric=metrics.Metrics(relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado,irrelevante_no_recuperado)
        print(f'P:{metric.precision_}')
        print(f'R:{metric.recall_}')
        print(f'F1:{metric.f1_}')
        print(f'Fal:{metric.fallout_}')
        
    for i in boolean_recovered.keys():
        recuperado=boolean_recovered[i]
        relevante=dic[i]
        relevante_recuperado=set(recuperado).intersection(set(relevante))
        relevante_no_recuperado=set(relevante).difference(set(recuperado))
        irrelevante_recuperado=set(recuperado).difference(set(relevante))
        irrelevante_no_recuperado=(set(doc_ids).difference(set(recuperado))).difference(set(relevante))
        print('   ')
        print(len(relevante_recuperado))
        print(len(relevante_no_recuperado))
        print(len(irrelevante_recuperado))
        print(len(irrelevante_no_recuperado))
        metric=metrics.Metrics(relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado,irrelevante_no_recuperado)
        print(f'P_B:{metric.precision_}')
        print(f'R_B:{metric.recall_}')
        print(f'F1_B:{metric.f1_}')
        print(f'Fal_B:{metric.fallout_}')

def submit_query():
    print("hola")

dataset=ir_datasets.load('cranfield')

docs={}
for doc in dataset.docs_iter():
    docs[doc.doc_id]=(doc.title,doc.text)

precomputed = precomputer.Precomputed(dataset=dataset)

boolean = BooleanModel(precomputed.corpus)
lsi=lsi_model.Lsi_model(precomputed.corpus)

# Crear la ventana principal

class BusquedaInterfaz(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Interfaz de Búsqueda")
        self.geometry("600x400")

        # Barra de búsqueda
        self.barra_busqueda = ttk.Entry(self)
        self.barra_busqueda.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Botón de búsqueda
        self.boton_buscar = ttk.Button(self, text="Buscar", command=self.buscar)
        self.boton_buscar.pack(side=tk.TOP, padx=10, pady=10)

        # Checkbox
        self.checkbox_frame = ttk.Frame(self)
        self.checkbox_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.booleano_var = tk.BooleanVar()
        self.booleano_checkbox = ttk.Checkbutton(self.checkbox_frame, text="Booleano", variable=self.booleano_var, command=self.seleccionar_checkbox_boolean)
        self.booleano_checkbox.pack(side=tk.LEFT)

        self.lsy_var = tk.BooleanVar()
        self.lsy_checkbox = ttk.Checkbutton(self.checkbox_frame, text="LSY", variable=self.lsy_var, command=self.seleccionar_checkbox_lsy)
        self.lsy_checkbox.pack(side=tk.LEFT)

        # Resultados de la búsqueda
        self.resultados_frame = ttk.Frame(self)
        self.resultados_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.resultados_scrollbar = ttk.Scrollbar(self.resultados_frame)
        self.resultados_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.resultados_listbox = tk.Listbox(self.resultados_frame, yscrollcommand=self.resultados_scrollbar.set)
        self.resultados_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.resultados_scrollbar.config(command=self.resultados_listbox.yview)

        # Eventos
        self.barra_busqueda.bind("<Return>", self.buscar)

        # Estilos y colores
        # self.configure(bg="#f0f0f0")
        # self.barra_busqueda.configure(bg="#ffffff", fg="#000000")
        # self.checkbox_frame.configure(bg="#f0f0f0")
        # self.resultados_frame.configure(bg="#ffffff")
        # self.resultados_listbox.configure(bg="#ffffff", fg="#000000")

        # Seleccionar un checkbox por defecto
        self.lsy_var.set(True)

    def buscar(self, event=None):
        texto_busqueda = self.barra_busqueda.get()
        # Aquí iría la lógica de búsqueda real, por ejemplo, consultar una base de datos o un motor de búsqueda.
        documents = []
        if self.lsy_var.get():
            documents = lsi.get_most_similar(texto_busqueda)
        else:
            documents = boolean.get_matching_docs(texto_busqueda)[:10]
        
        resultados = map(lambda doc_id: {'titulo':docs[str(doc_id)][0],'contenido':docs[str(doc_id)][1]},documents)
        
        # Ejemplos de prueba de resultados de búsqueda
        # resultados = [
        #     {"titulo": texto_busqueda, "contenido": "Este es el contenido del resultado 1"},
        #     {"titulo": "Resultado 2", "contenido": "Este es el contenido del resultado 2"},
        #     {"titulo": "Resultado 3", "contenido": "Este es el contenido del resultado 3"},
        # ]

        self.mostrar_resultados(resultados)

    def mostrar_resultados(self, resultados):
        self.resultados_listbox.delete(0, tk.END)
        for resultado in resultados:
            self.resultados_listbox.insert(tk.END, resultado['titulo'])
            self.resultados_listbox.insert(tk.END, resultado['contenido'][:100])
            self.resultados_listbox.insert(tk.END, '')

    def seleccionar_checkbox_boolean(self):
        # Asegurarse de que solo un checkbox esté seleccionado
        if self.booleano_var.get():
            self.lsy_var.set(False)
        else:
            self.booleano_var.set(True)
    def seleccionar_checkbox_lsy(self):
        # Asegurarse de que solo un checkbox esté seleccionado
        if self.lsy_var.get():
            self.booleano_var.set(False)
        else:
            self.lsy_var.set(True)

if __name__ == "__main__":
    interfaz = BusquedaInterfaz()
    interfaz.mainloop()

