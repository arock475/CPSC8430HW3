In this project we were tasked with fine-tuning a pre-trained BERT model in order to allow it to work better on extractive question answering. BERT is a widely used machine learning model which is primarily used in Natural Language Processing. Typically BERT is used to select the word which is hidden in a sentence using the context around it. In this project we were tasked with taking a pre-trained BERT model from Hugging Face and to finetune it so that it will function in extractive question answering. Extractive question answering is when, along with the question, a passage or section of text is given which contains the answer to said question. Thus the model is able to answer the question by giving the indexes of the start and end of the answer. 
I utilized Google's google-bert/bert-base-uncased (https://huggingface.co/google-bert/bert-base-uncased) model for this project and fine-tuned it using the Spoken-Squad dataset (https://github.com/chiahsuan156/Spoken-SQuAD).
In order to run the finetuning program simply type:
```
python .\Finetuning.py
```
or possibly
```
python3 .\Finetuning.py
```
depending on your computer's setup. Ensure you have the Transformers library installed as well as Torch. 

If running the model ensure you have a folder named SavedModel (case sensitive) to save the model to otherwise the code will fail.

In order to evaluate the model after finetuning it simply run 
```
python .\eval.py
```
or possibly
```
python3 .\eval.py
```
Inside of the file ensure to change the model which is being evaluated.
