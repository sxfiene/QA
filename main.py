from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

pipe_bert = pipeline("question-answering", model="DiDiR6/Bert-QA")
pipe_roberta = pipeline("question-answering", model="DiDiR6/Bert-QA")

tokenizer_t5 = AutoTokenizer.from_pretrained("DiDiR6/T5-QA")
model_t5 = AutoModelForQuestionAnswering.from_pretrained("DiDiR6/T5-QA")

app = FastAPI()


class Question(BaseModel):
    content: str
    context: str
    source: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/question/")
async def answer(question: Question):
    if question.source == "bert":
        print("BERT")
        content = question.content.strip()
        result = pipe_bert(question=content, context=question.context)
        return {"role": "assistant", "content": result["answer"]}

    if question.source == "roberta":
        print("RoBERTa")
        content = question.content.strip()
        result = pipe_roberta(question=content, context=question.context)
        return {"role": "assistant", "content": result["answer"]}

    if question.source == "t5":
        print("T5")
        content = question.content.strip()

        encoding = tokenizer_t5.encode_plus(content, question.context, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        outputs = model_t5(input_ids, attention_mask=attention_mask)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits

        all_tokens = tokenizer_t5.convert_ids_to_tokens(input_ids[0])
        result = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        return {"role": "assistant", "content": result}
