import spacy
import logging
import argparse
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='file on DataFrame format')
    parser.add_argument('--path_save', type=str, required=True, help='path to save file')
    parser.add_argument('--spacy_model', type=str, default='ru_core_news_lg', help='spacy model name')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose NERs')
    args, _ = parser.parse_known_args()
    return args


def skip_not_need(x: str) -> list:
    r"""
    we have drop rows because:
       - grettings&names and organization in first 3 row
       - goodbyes ends 3 row
    we get 6 row for each manager
    """
    agg_val = x.tolist()
    return agg_val[:3] + agg_val[-3:]


def ner_ruler(name_model: str) -> spacy.lang:
    nlp = spacy.load(name_model)
    ruler = nlp.add_pipe('entity_ruler', config={"overwrite_ents": True})
    patterns = [
        {"label": "find_name", "pattern": [
            {'LEMMA': 'меня', "OP": "+"},
            {'LEMMA': 'звать', "OP": "?"},
            {'ENT_TYPE': 'PER', "OP": "+"}]},
        {'label': 'find_org', 'pattern': [
            {"LOWER": {"REGEX": r"(компан|фирм)"}, "OP": "+"},
            {"LOWER": {"REGEX": r"\w+"}}]},
        {'label': 'find_greetings', 'pattern': [
            {"LEMMA": {"IN": ['здравствовать', 'привет', 'добрый']}, "OP": "+"},
            {"LOWER": {"REGEX": r"(день|утро|вечер)", "OP": "?"}}]},
        {'label': 'find_goodbyes', 'pattern': [
            {"LEMMA": {"IN": ['до', 'всего', 'весь'], "OP": "+"}},
            {"LEMMA": {"IN": ['свидание', 'хороший', 'добрый'], "OP": "+"}}]}
    ]
    ruler.add_patterns(patterns)
    return nlp


def parse_same_ner(
    data: pd.DataFrame,
    model_name: str,
    path_save_file: str,
    verbose: bool = False
) -> None:
    r"""
    data: pd.DataFrame, dataframe where need find ner
    model_name: str, spacy pretrain model name
    path_save_file: str, path to save file with name file
    verbose: bool, print find ner
    return: None
    Explain:
        we add index to text
        add new columns
        aggregate text by manager, and take only six rows
        loads spacy model and add rule to ner
        loop data:
            split each text by '_' and get index position on data and text
            find ner and add to data by index
    """
    data['index_text'] = [str(k) + '_' + v for k, v in zip(range(len(data.text.values)), data.text.values)]
    new_col = ['find_name', 'find_org', 'find_goodbyes', 'find_greetings', 'greetings_goodbyes']
    list_text = data[data.role == 'manager'].groupby(['dlg_id'])['index_text'].apply(skip_not_need)
    model = ner_ruler(model_name)
    data[new_col] = [False] * len(new_col)
    pbar = tqdm(desc="Find ner's ...", total=int(len(list_text)), disable=True if verbose else False)
    for i in range(len(list_text)):
        six_task = set()
        for doc in model.pipe(list_text[i]):
            idx, text = doc.text.split('_')
            idx = int(idx)
            doc = model(text)
            for ent in doc.ents:
                if ent.label_ == 'find_name':
                    data.loc[idx, new_col[0]] = ent.text.split()[-1]
                if ent.label_ == 'find_org':
                    data.loc[idx, new_col[1]] = ent.text
                if ent.label_ == 'find_goodbyes':
                    data.loc[idx, new_col[2]] = ent.text
                    six_task.add(1)
                if ent.label_ == 'find_greetings':
                    data.loc[idx, new_col[3]] = ent.text
                    six_task.add(2)
                if verbose:
                    logger.info(f"Find NER in row: {idx}, text: {ent.text}, label: {ent.label_}")
        if sum(six_task) == 3:
            data.loc[idx, new_col[-1]] = True
        pbar.update(1)
    pbar.close()
    data.drop('index_text', axis=1).to_csv(path_save_file, index=False)
    logger.info(f'File saved on: {path_save_file}')


if __name__ == "__main__":
    assert spacy.__version__ >= '3.4.0', f'error version spacy, loaded version is {spacy.__version__}'
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    logging.getLogger('pymorphy2').setLevel(logging.CRITICAL)
    logger.info('Start script')
    data = pd.read_csv(args.path)
    parse_same_ner(data, args.spacy_model, args.path_save, args.verbose)
