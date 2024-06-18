from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def ask_model(question, model, tokenizer):
    # Кодирование вопроса
    encoded_input = tokenizer.encode(question, return_tensors='pt', add_special_tokens=True)
    # Генерация ответа моделью
    output = model.generate(encoded_input, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    # Декодирование ответа в текст
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Пример вопроса
question = "the capital of france"

# Получение ответа от модели
answer = ask_model(question, model, tokenizer)
print(answer)