from typing import List, Tuple, Dict
from tqdm import tqdm
import copy
from collections import defaultdict

import scipy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from lib.bm25 import BM25Vectorizer


class WorldTreeKB:
    """The knowledge base used to store scientific facts and return relevant facts given a query.

    The KB first takes a list of scientific facts as input. Then, it calculates the bm25 parameters on top
    of these facts. Finally, given a query, it returns the relevant facts assicuated with this query.

        For example: 
            facts = [
                {"id": 1, "original_fact": "an apples is a kind of fruit", "lemmatized_fact": "apple be kind of fruit"},
                {"id": 2, "original_fact": "a girl is a kind of human", "lemmatized_fact": "girl be kind of human"},]
            query = "what is an apple?"

        it should return the relevant facts:
            relevant_facts = [
                {"id": 1, "original_fact": "an apples is a kind of fruit", "lemmatized_fact": "apple be kind of fruit"}
            ]


    Attributes:    
        facts: `List`, required 
            A list of facts. A fact is a user-defined dict object that contains the information
            for a scientific fact. It should contain the required fields "id" and "lemmatized_fact".
        id_to_fact: `Dict`, optional (default=`False`)  
            A dict that maps the fact id to the associated fact.
        ranking_function: `Object`, optional (default=`BM25`)  
            The ranking function for the knowledge base.

    """

    def __init__(self,
                 facts: List[Dict[str, str]],
                 rank_func: str = "BM25",
                 corpus_to_fit=None):
        """
        Args:
            facts: `List`, required 
                See `facts` in the Attribute docstring
            rank_func: `str`, optional (default=`BM25`)  
                A string to assign the ranking function for this knowledge base.
            corpus_to_fit: `List[str]`, optional (default=None)  
                A list of strings used to feed to the ranking function and tune its parameters.
                If `corpus_to_fit` is None, the ranking function will fit on `self.facts` by default.
        """
        self.facts = facts
        self.id_to_fact = {}
        self.ranking_function = None

        # Initialize the id_to_fact mapping
        for fact in self.facts:
            self.id_to_fact[fact["id"]] = fact

        # Initialize the ranking function
        # TODO: add another ranking function?
        if rank_func == "BM25":
            self.ranking_function = BM25Vectorizer()

        # Fit the ranking function and transform the facts
        # TODO: if the corpus to fit is None
        self.ranking_function.fit(corpus_to_fit)
        self._transformed_facts = self.ranking_function.transform(
            [x["lemmatized_fact"] for x in self.facts])

    def query_relevant_facts(self, query: str, topk: int = None):
        """
        Args
            query: `str`, required
                The query string.
            topk: `int`, optional
                The number of top candidates to be considered.

        Returns
            relevant_facts: `List[Dict]`
                A list of relevant facts.
            id_to_score: `Dict`
                The dict that maps the ids of the relevant facts to their score.
        """
        # Calculate cosine similarity
        transformed_query = self.ranking_function.transform([query])
        # Shape: 1*num_facts -> facts
        similarities = cosine_distances(
            transformed_query, self._transformed_facts
        )[0]

        # Get topk relevant facts
        rank = np.argsort(similarities)  # Descending order
        if topk:
            rank = rank[:topk]
        relevant_facts = []
        for index in rank:
            fact = copy.deepcopy(self.facts[index])
            score = 1 - similarities[index]
            fact["relevance_score"] = score
            relevant_facts.append(fact)

        # Get id_to_score mapping
        id_to_score = {}
        for fact in relevant_facts:
            idx = fact["id"]
            id_to_score[idx] = fact["relevance_score"]

        return relevant_facts, id_to_score


class QuestionKB:
    """The old question knowledge base that aims to return the similar questions and their explanations given a query.

    Attributes:    
        questions: `List`, required 
            A list of question. A question is a user-defined dict object that contains the information
            for a scientific question. It should contain the required fields "id" and "lemmatized_question".
        ranking_function: `Object`, optional (default=`BM25`)  
            The ranking function for the knowledge base.

    """

    def __init__(self,
                 questions: List[Dict[str, str]],
                 rank_func: str = "BM25",
                 corpus_to_fit=None):
        """
        Args:
            questions: `List`, required 
                See `questions` in the Attribute docstring
            rank_func: `str`, optional (default=`BM25`)  
                A string to assign the ranking function for this knowledge base.
            corpus_to_fit: `List[str]`, optional (default=None)  
                A list of strings used to feed to the ranking function and tune its parameters.
                If `corpus_to_fit` is None, the ranking function will fit on `self.questions` by default.
        """
        self.questions = questions
        self.ranking_function = None

        # Initialize the ranking function
        # TODO: add another ranking function?
        if rank_func == "BM25":
            self.ranking_function = BM25Vectorizer()

        # Fit the ranking function and transform the questions
        # TODO: if the corpus to fit is None
        self.ranking_function.fit(corpus_to_fit)
        self._transformed_questions = self.ranking_function.transform(
            [x["lemmatized_question"] for x in self.questions])

    def query_similar_questions(self, query: str, topk):
        """Find the similar old exam questions.

        Args:
            query: `str`, required
                The query string.
            topk: `int`, optional
                The number of top candidates to be considered.

        Returns:
            similar_questions: `List[Dict]`
                A list of similar questions.
        """
        # Calculate cosine similarity
        transformed_query = self.ranking_function.transform([query])
        # Shape: 1*num_questions -> questions
        similarities = cosine_distances(
            transformed_query, self._transformed_questions
        )[0]

        # Get topk question ids
        rank = np.argsort(similarities)  # Descending order
        if topk:
            rank = rank[:topk]

        # Get the similar questions
        similar_questions = []
        for index in rank:
            question = copy.deepcopy(self.questions[index])
            score = 1 - similarities[index]
            question["score"] = score
            similar_questions.append(question)

        return similar_questions

    def query_facts_from_similar_questions(self,
                                           query: str,
                                           topk_questions: int = None,
                                           topk_facts: int = None):
        """Query the relevant facts from similar old exam questions.

        Note that the unification score for each fact is accumulated across questions. 
        That is, if a fact appears in 3 questions, its overall unification score is the sum of the
        respective score in these questions.

        Args:
            query: `str`, required
                The query string.
            topk_questions: `int`, optional
                The number of top similar questions to be considered.
            topk_facts: `int`, optional
                The number of facts to be considered.

        Returns:
            relevant_facts: `List[Dict]`
                A list of relevant facts.
            id_to_score: `Dict`
                The dict that maps the ids of the relevant facts to their unification score.
        """
        similar_questions = self.query_similar_questions(query, topk_questions)

        # Accumulate the unification scores for each fact across questions
        unification_scores = defaultdict(lambda: 0)
        for question in similar_questions:
            for fact in question['_explanation']:
                unification_scores[fact] += question["score"]

        # Get topk fact ids
        rank = sorted(unification_scores,
                      key=unification_scores.get, reverse=True)
        if topk_facts:
            rank = rank[:topk_facts]

        # Get the relevant facts and a id-to-score mapping
        relevant_facts = []
        id_to_score = {}
        for idx in rank:
            score = unification_scores[idx]
            relevant_facts.append(
                {"id": idx, "unification_score": score}
            )
            id_to_score[idx] = score

        return relevant_facts, id_to_score
