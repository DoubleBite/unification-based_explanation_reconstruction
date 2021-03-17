from tqdm import tqdm
import msgpack
import nltk
from nltk.corpus import stopwords

from lib.knowledge_base import WorldTreeKB, QuestionKB
from lib.utility import preprocess_fact, WorldTreeLemmatizer, preprocess_question, preprocess_dev_question

# Parameters
K = 5000  # relevance facts limit
Q = 100  # similar questions limit
QK = 85  # unification facts limit
weights = [0.83, 0.17]  # relevance and unification score weigths


# Load the table store
with open("data/cache/table_store.mpk", "rb") as f:
    tablestore = msgpack.unpackb(f.read(), raw=False)

# Load the training set
with open("data/cache/eb_train.mpk", "rb") as f:
    train_set = msgpack.unpackb(f.read(), raw=False)

# Load the dev set
with open("data/cache/eb_dev.mpk", "rb") as f:
    dev_set = msgpack.unpackb(f.read(), raw=False)


facts = []
questions = []
corpus_to_fit = []

lemmatizer = WorldTreeLemmatizer()


# Prepare the facts and questions
for idx, ts in tqdm(tablestore.items()):

    fact = ts["_sentence_explanation"]
    lemmatized_fact = preprocess_fact(fact, lemmatizer)

    corpus_to_fit.append(" ".join(lemmatized_fact))
    facts.append({
        "id": idx,
        "original_fact": " ".join(fact),
        "lemmatized_fact": " ".join(lemmatized_fact)
    })


for q_id, question in tqdm(train_set.items()):

    question_text = question["_question"]
    if question["_answerKey"] in question["_choices"]:
        choice = question["_choices"][question["_answerKey"]]
    else:
        choice = ""

    lemmatized_question = preprocess_question(
        question_text, choice, lemmatizer)

    corpus_to_fit.append(lemmatized_question)
    questions.append({
        "id": q_id,
        "original_question": question_text,
        "lemmatized_question": lemmatized_question,
        "_explanation": question["_explanation"]
    })


# Initialize the fact and question knowledge bases
worldtree_kb = WorldTreeKB(facts, corpus_to_fit=corpus_to_fit)
quesiton_kb = QuestionKB(questions, corpus_to_fit=corpus_to_fit)


# Retrieve relevant facts for dev set questions
results = []
for q_id, question in tqdm(dev_set.items()):

    # Preprocess the question text and choice
    question_text = question["_question"]
    if question["_answerKey"] in question["_choices"]:
        choice = question["_choices"][question["_answerKey"]]
    else:
        choice = ""
    lemmatized_question = preprocess_dev_question(
        question_text, choice, lemmatizer)

    # Get the relevant facts and their scores from factKB and questionKB
    _, relevance_scores = worldtree_kb.query_relevant_facts(
        lemmatized_question, K)
    _, unification_scores = quesiton_kb.query_facts_from_similar_questions(
        lemmatized_question, Q, QK)

    # Merge the scores
    # We iterate through each fact and check whether it has a relevance or unification score
    combined_scores = {}
    for t_id, ts in tablestore.items():
        if t_id in relevance_scores and t_id in unification_scores:
            combined_scores[t_id] = (weights[0] * relevance_scores[t_id]) + (
                weights[1] * unification_scores[t_id]
            )
        elif t_id in relevance_scores:
            combined_scores[t_id] = weights[0] * relevance_scores[t_id]
        elif t_id in unification_scores:
            combined_scores[t_id] = weights[1] * unification_scores[t_id]
        else:
            combined_scores[t_id] = 0

    # Sort the scores
    for fact in sorted(combined_scores, key=combined_scores.get, reverse=True):
        result = q_id + "\t" + fact
        results.append(result)


# Write out the result
with open("prediction.txt", "w") as f:
    for result in results:
        print(result, file=f)
