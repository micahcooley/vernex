"""
Code-focused training data for Vernex 500M.
70% Code, balanced debug/chat. Includes uncertainty handling.
"""
import random
import os

# === MINIMAL CHAT ===
GREETINGS = ["Hi", "Hello", "Hey", "Hi there"]
GREETING_RESPONSES = ["Hi! What code do you need?", "Hello! Show me the code.", "Hey! What's the task?"]

# === CODE SAMPLES ===
CLASSES = ["Delay", "Reverb", "Compressor", "EQ", "Limiter", "Filter", "Phaser", "Chorus", "Distortion"]
JUCE_TYPES = ["juce::AudioBuffer<float>", "juce::MidiBuffer", "juce::IIRFilter", "juce::ADSR"]

def gen_greeting():
    return f"<|im_start|>user\n{random.choice(GREETINGS)}<|im_end|>\n<|im_start|>assistant\n{random.choice(GREETING_RESPONSES)}<|im_end|>\n"

def gen_code_request():
    cls = random.choice(CLASSES)
    return f"""<|im_start|>user
Write a {cls} class in C++<|im_end|>
<|im_start|>assistant
<thought>User wants a {cls} class. Implement basic process loop.</thought>
```cpp
class {cls} {{
    float param = 0.5f;
public:
    void process(float* data, int n) {{
        for (int i = 0; i < n; ++i) {{
            data[i] *= param; // Basic {cls} logic
        }}
    }}
}};
```<|im_end|>
"""

def gen_juce_snippet():
    return f"""<|im_start|>user
How do I clear a JUCE AudioBuffer?<|im_end|>
<|im_start|>assistant
Use the `clear()` method:
```cpp
void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {{
    buffer.clear();
}}
```<|im_end|>
"""

def gen_search_uncertainty():
    unknowns = ["latest version of FabFilter Pro-Q", "Logic Pro 11 release notes", "new JUCE 8 features", "upcoming audio tech"]
    u = random.choice(unknowns)
    return f"""<|im_start|>user
What are the {u}?<|im_end|>
<|im_start|>assistant
<thought>I don't have this in my recent training data. I should search.</thought>
I'm not exactly sure about the latest details on {u}. Let me check for you:
[SEARCH: {u}]<|im_end|>
"""

def gen_fim():
    cls = random.choice(CLASSES)
    return f"<PRE>void {cls}::process(float* d, int n) {{\n    <SUF>\n}}<MID>for(int i=0;i<n;++i)d[i]*=0.5f;"

def generate_corpus(filepath, entries=15000):
    print(f"Generating {entries} code-heavy examples...")
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(entries):
            r = random.random()
            # 70% Code Focus
            if r < 0.70:
                f.write(gen_code_request() if random.random() < 0.7 else gen_juce_snippet())
            # 15% FIM
            elif r < 0.85:
                f.write(gen_fim())
            # 10% Search/Uncertainty
            elif r < 0.95:
                f.write(gen_search_uncertainty())
            # 5% Chat
            else:
                f.write(gen_greeting())
            
            if i % 3000 == 0:
                print(f"  {i}/{entries}...")
    print("Done.")

if __name__ == "__main__":
    os.makedirs("c:/vernex/data", exist_ok=True)
    generate_corpus("c:/vernex/data/audio_corpus.txt", entries=15000)
