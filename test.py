from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer_t5 = AutoTokenizer.from_pretrained("DiDiR6/T5-QA")
model_t5 = AutoModelForQuestionAnswering.from_pretrained("DiDiR6/T5-QA")

question = "Qui est Emmanuel Macron ?"
context = "Emmanuel Macron est le président de la France"

encoding = tokenizer_t5.encode_plus(question, context, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

outputs = model_t5(input_ids, attention_mask=attention_mask)
start_scores, end_scores = outputs.start_logits, outputs.end_logits

all_tokens = tokenizer_t5.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])

print('Question:', question)
print(answer)
