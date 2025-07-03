
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Orbina/Orbita-v0.1",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Orbina/Orbita-v0.1")

prompt = "Python'da ekrana 'Merhaba Dünya' nasıl yazılır?"
messages = [
    {"role": "system", "content": "Sen, Orbina AI tarafından üretilen ve verilen talimatları takip ederek en iyi cevabı üretmeye çalışan yardımcı bir yapay zekasın."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    temperature=0.3,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
    repetition_penalty=1,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
