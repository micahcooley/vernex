from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os

PRE_TOKEN = "<PRE>"
SUF_TOKEN = "<SUF>"
MID_TOKEN = "<MID>"

def train_vernex_tokenizer(save_path="c:/vernex/model/tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=32000, 
        special_tokens=["<PRE>", "<SUF>", "<MID>", "<|im_start|>", "<|im_end|>", "<|tool_result|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    files = ["c:/vernex/data/audio_corpus.txt"]
    if not os.path.exists(files[0]):
        print("Corpus not found!")
        return

    print(f"Training tokenizer on {len(files)} files...")
    tokenizer.train(files, trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_vernex_tokenizer()
