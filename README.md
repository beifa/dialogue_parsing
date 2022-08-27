## Script to find NER in text file

NERs:
- greeting - where the manager said hello.
- hello - manager introduced himself.
- names - manager name.
- company - company name.
- goodbye - manager said goodbye.

PS:

*Check the requirement for the manager: “In each dialogue, it is imperative to say hello and say goodbye to the client”*

## Instruction to Run:

```python
>>> script/parsing_sacript.py
    -- path                 # path to data to finds ner
    -- path_save            # path to save
    -- model_name           # spacy model to loads, default 'ru_core_news_lg'
    -- verbose              # print results, default false
```
or check notebook version script ...