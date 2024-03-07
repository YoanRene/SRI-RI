from sympy import sympify, to_dnf, Not, And, Or
from preprocess import preprocess
class BooleanModel:
    def __init__(self, corpus):
        # self.nlp = nlp
        # self.tokenized_docs = tokenized_docs
        # self.dictionary = dictionary
        self.corpus = corpus
    def query_to_dnf(self, query):
        """
        Convierte y simplifica la expresión lógica a su forma normal disyuntiva.

        Arg:
            - query (str): Consulta de entrada.

        Return:
            query_dnf(str): Consulta expresada en forma normal disyuntiva.
        """
        # #Tokenizamos la query
        # tokens = [token.lemma_.lower() for token in self.nlp(query) if token.is_alpha or token.text == ")" or token.text == "("]
        #La procesamos para usar los simbolos logicos de symplify
        preprocessed_query = preprocess(query,True)
        # processed_query = preprocessed_query.replace("and", "&").replace("or", "|").replace("not", "~")
        processed_query = preprocessed_query.upper()

        #Creamos la forma normal disyuntiva
        query_expr = sympify(processed_query, evaluate=False,convert_xor=False)
        query_dnf = to_dnf(query_expr)

        return query_dnf

    def get_matching_docs(self,query):
        query_dnf = self.query_to_dnf(query)
        return self._get_matching_docs(query_dnf)

    def _get_matching_docs(self, query_dnf):
        """
        Obtiene documentos que coinciden con la forma normal disyuntiva (DNF) proporcionada.

        Args:
            - query_dnf (str): Consulta en forma normal disyuntiva (DNF).

        Returns:
            list: Lista de documentos que satisfacen la consulta DNF dada.
        """
        # Función para verificar si un documento satisface una componente conjuntiva de la consulta
        matching_documents = []
        #Verificamos cada documento del corpus
        for doc,doc_id in self.corpus:
            #Variable que cambia a false en el momento que no satisfasca una componente
            satisfied = False
            #Verificamos cada componente
            if query_dnf.is_Atom:
                satisfied = self.satisfies_conjunctive_component(doc,query_dnf)
            elif  isinstance(query_dnf, And):
                satisfied = all(self.satisfies_conjunctive_component(doc, conjunct) for conjunct in query_dnf.args)
            else:      
                for component in query_dnf.args:
                    if isinstance(component, And):
                        if all(self.satisfies_conjunctive_component(doc, conjunct) for conjunct in component.args):
                            satisfied = True
                            break
                    else:
                        if self.satisfies_conjunctive_component(doc, component):
                            satisfied = True
                            break
            if satisfied:
                matching_documents.append(doc_id)
        return matching_documents

    def satisfies_conjunctive_component(self,doc, component):
        """
        Verifica si el documento satisface la componente
        Args:
            - doc : Documento a evaluar
            - component: Componente conjuntiva a satisfacer
        Returns:
            bool: True if the document satisfies the conjunctive component, False otherwise.
        """
        if isinstance(component, Not):
            if component.name.lower() in doc:
                return False
        else:
            if component.name.lower() not in doc:
                return False
        return True