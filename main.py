import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# ---- 1. Define Idioms Dataset ---- #
idioms_data = [
    {"hi": "रस्सी जल गयी, बल नहीं गया", "en": "Even after one hits rock bottom, their arrogance remains unchanged."},
    {"hi": "नाच ना जाने आँगन टेढ़ा", "en": "A bad workman blames his tools."},
    {"hi": "ना नौ मन तेल होगा, ना राधा नाचेगी", "en": "If the task is never completed, celebration is meaningless."},
    {"hi": "उल्टा चोर कोतवाल को डाँटे", "en": "The guilty blames the innocent."},
    {"hi": "आ बैल मुझे मार", "en": "Inviting trouble unnecessarily."},
    {"hi": "घर की मुर्गी दाल बराबर", "en": "Familiar things are often undervalued."},
    {"hi": "एक अनार सौ बीमार", "en": "Too many people vying for one opportunity."},
    {"hi": "बंदर क्या जाने अदरक का स्वाद", "en": "Someone who doesn't appreciate something valuable."},
    {"hi": "अंधेर नगरी चौपट राजा", "en": "A place where there's no rule and the leader is incompetent."},
    {"hi": "धीरे-धीरे रे मना, धीरे सब कुछ होय", "en": "Patience is the key to achieving everything."},
    {"hi": "जहाँ चाह वहाँ राह", "en": "Where there’s a will, there’s a way."},
    {"hi": "ओखली में सिर दिया, तो मूसल से क्या डर", "en": "If you've taken the risk, don't fear the consequences."},
    {"hi": "अपने मुँह मियाँ मिठू", "en": "To brag about oneself."},
    {"hi": "आसमान से गिरा, खजूर में अटका", "en": "Out of one problem, into another."},
    {"hi": "दाल में कुछ काला है", "en": "Something is fishy."},
    {"hi": "जिसकी लाठी उसकी भैंस", "en": "Might is right."},
    {"hi": "ऊँट के मुँह में जीरा", "en": "A drop in the ocean."},
    {"hi": "खिसियानी बिल्ली खंभा नोचे", "en": "A frustrated person lashes out irrationally."},
    {"hi": "जो गरजते हैं वो बरसते नहीं", "en": "Those who boast often don't deliver."},
    {"hi": "साँप भी मर जाए और लाठी भी न टूटे", "en": "A solution where both parties are satisfied."},
    {"hi": "बिल्ली के भाग्य से छींका टूटा", "en": "A rare stroke of luck."},
    {"hi": "थोथा चना बाजे घना", "en": "Empty vessels make the most noise."},
    {"hi": "निंदक नियरे राखिए", "en": "Keep your critics close, they help you improve."},
    {"hi": "करेला ऊपर से नीम चढ़ा", "en": "Something already bad just got worse."},
    {"hi": "अंधे के हाथ बटेर लगना", "en": "A blind man finding a quail – extreme luck."},
    {"hi": "पग-पग पर बिघ्न आते हैं", "en": "Trouble at every step."},
    {"hi": "दो नावों की सवारी ठीक नहीं", "en": "You can’t ride two boats at once."},
    {"hi": "जिसका काम उसी को साजे", "en": "Only the skilled should handle their task."},
    {"hi": "गधे को घोड़ा समझना", "en": "To overestimate someone unworthy."},
    {"hi": "मुँह में राम, बगल में छुरी", "en": "To pretend goodness while planning harm."}
]

# ---- 2. Convert to Hugging Face Dataset ---- #
dataset = Dataset.from_list(idioms_data)

# ---- 3. Load Model and Tokenizer ---- #
model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# ---- 4. Preprocessing ---- #
def preprocess(example):
    model_inputs = tokenizer(example["hi"], max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(example["en"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess)

# ---- 5. Training ---- #
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

with st.spinner("Training model... (this may take time on first run)"):
    trainer.train()

# ---- 6. Define Translation Function ---- #
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids
    output = model.generate(inputs, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---- 7. Streamlit UI ---- #
st.set_page_config(page_title="Hindi Idioms Translator", layout="centered")
st.title("📚 Hindi Idioms Translator")
st.write("This app fine-tunes a MarianMT model to translate Hindi idioms to English.")

user_input = st.text_area("✍️ Enter a Hindi idiom:", height=100, placeholder="Example: रस्सी जल गयी, बल नहीं गया")

if st.button("🔄 Translate"):
    if user_input.strip():
        output = translate(user_input)
        st.success("✅ Translation:")
        st.write(f"**{output}**")
    else:
        st.warning("⚠️ Please enter a valid Hindi idiom.")
