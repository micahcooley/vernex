import random
import os

# Vocabulary for variety
ADJ = ["fast", "slow", "buggy", "clean", "messy", "old", "new", "complex", "simple", "broken", "good", "bad"]
NOUNS = ["code", "plugin", "module", "function", "class", "buffer", "signal", "sample", "filter", "effect", "audio"]
VERBS = ["fix", "check", "review", "optimize", "refactor", "debug", "test", "improve", "write", "create", "build"]
GREETINGS = ["Hi", "Hello", "Hey", "Yo", "Morning", "Evening", "Sup", "Hi there", "Hey there", "Whats up"]
RESPONSES = ["What do you need?", "Go ahead.", "Ready.", "Listening.", "Yes?", "Here.", "Whats the task?", "Show me."]
CLASSES = ["Delay", "Reverb", "Compressor", "EQ", "Limiter", "Phaser", "Chorus", "Flanger", "Distortion", "Gate"]
ISSUES = ["crackling", "glitching", "stuttering", "dropping", "distorting", "clipping", "popping", "buzzing"]
CAUSES = ["allocation", "blocking call", "mutex", "file IO", "vector resize", "new operator", "lock contention"]

def gen_greeting():
    return f"<|im_start|>user\n{random.choice(GREETINGS)}<|im_end|>\n<|im_start|>assistant\n{random.choice(RESPONSES)}<|im_end|>\n"

def gen_qa():
    topic = random.choice(["audio processing", "DSP", "JUCE framework", "C++ optimization", "plugin development", "latency", "buffer management", "filter design", "real-time audio", "synthesis"])
    fact = random.choice([f"{topic} is important.", f"Focus on {random.choice(ADJ)} {random.choice(NOUNS)}.", f"Learn {topic} well."])
    return f"<|im_start|>user\nTell me about {topic}<|im_end|>\n<|im_start|>assistant\n{fact}<|im_end|>\n"

def gen_code():
    cls = random.choice(CLASSES)
    gain = round(random.uniform(0.1, 1.0), 2)
    return f"""<|im_start|>user
Write a {cls} class<|im_end|>
<|im_start|>assistant
<thought>User wants {cls}. Simple and complete.</thought>
```cpp
class {cls} {{
    float gain = {gain}f;
public:
    void process(float* d, int n) {{ for(int i=0;i<n;++i) d[i] *= gain; }}
}};
```<|im_end|>
"""

def gen_debug():
    issue = random.choice(ISSUES)
    cause = random.choice(CAUSES)
    return f"""<|im_start|>user
My plugin is {issue}<|im_end|>
<|im_start|>assistant
<thought>Audio {issue}. Realtime violation likely.</thought>
Check for {cause} in audio thread. Move to prepareToPlay().<|im_end|>
"""

def gen_fim():
    cls = random.choice(CLASSES)
    return f"<PRE>void {cls}::process(float* d, int n) {{\n    <SUF>\n}}<MID>for(int i=0;i<n;++i)d[i]*={round(random.uniform(0.1,1.0),2)}f;"

def generate_corpus(filepath, entries=8000):
    print(f"Generating {entries} examples...")
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(entries):
            r = random.random()
            if r < 0.25: f.write(gen_greeting())
            elif r < 0.45: f.write(gen_qa())
            elif r < 0.65: f.write(gen_code())
            elif r < 0.85: f.write(gen_debug())
            else: f.write(gen_fim())
            if i % 1000 == 0: print(f"  {i}/{entries}...")
    print("Done.")

if __name__ == "__main__":
    os.makedirs("c:/vernex/data", exist_ok=True)
    generate_corpus("c:/vernex/data/audio_corpus.txt", entries=8000)
