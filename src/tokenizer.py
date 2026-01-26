from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from pathlib import Path

# Resolve paths relative to project root
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

PRE_TOKEN = "<PRE>"
SUF_TOKEN = "<SUF>"
MID_TOKEN = "<MID>"

def train_vernex_tokenizer(save_path=None):
    if save_path is None:
        save_path = MODEL_DIR / "tokenizer.json"
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=32000, 
        special_tokens=["<PRE>", "<SUF>", "<MID>", "<|im_start|>", "<|im_end|>", "<|tool_result|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Find training data - prefer new enhanced data
    corpus_files = []
    enhanced_corpus = DATA_DIR / "cpp_juce_skia_corpus.txt"
    basic_corpus = DATA_DIR / "audio_corpus.txt"
    
    if enhanced_corpus.exists():
        corpus_files.append(str(enhanced_corpus))
    if basic_corpus.exists():
        corpus_files.append(str(basic_corpus))
    
    if not corpus_files:
        print(f"No corpus found in {DATA_DIR}! Run data_gen.py first.")
        return

    print(f"Training tokenizer on {len(corpus_files)} file(s)...")
    tokenizer.train(corpus_files, trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    MODEL_DIR.mkdir(exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_vernex_tokenizer()
