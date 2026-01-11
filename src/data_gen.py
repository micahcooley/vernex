"""
Diverse training data generation for Vernex.
Goal: Model should know when to debug vs when to chat casually.
"""
import random
import os

# === GREETING VARIATIONS ===
GREETINGS = [
    "Hi", "Hello", "Hey", "Yo", "Hi there", "Hello there", "Hey there",
    "Good morning", "Good afternoon", "Good evening", "Morning", "Evening",
    "Sup", "What's up", "Howdy", "Greetings", "Hey Vernex", "Hi Vernex",
    "Hello Vernex", "Yo Vernex", "Hey buddy", "Hello friend", "Whats good"
]

GREETING_RESPONSES = [
    "Hi! What can I help with?",
    "Hello! How can I assist?",
    "Hey! What do you need?",
    "Hi there. What's the task?",
    "Hello! Ready to code.",
    "Hey! What are we building?",
    "Hi. Show me what you're working on.",
    "Hello! Let me know what you need."
]

# === SIMPLE Q&A ===
SIMPLE_QA = [
    ("What is JUCE?", "JUCE is a C++ framework for audio applications and plugins."),
    ("What is latency?", "Latency is the delay between input and output in audio systems."),
    ("What is a buffer?", "A buffer is temporary storage for audio samples during processing."),
    ("What is sample rate?", "Sample rate is how many audio samples are captured per second."),
    ("What is bit depth?", "Bit depth determines the resolution of each audio sample."),
    ("What is DSP?", "DSP stands for Digital Signal Processing."),
    ("What is a DAW?", "DAW stands for Digital Audio Workstation."),
    ("What is VST?", "VST is a plugin format created by Steinberg."),
    ("What is AU?", "AU (Audio Unit) is Apple's plugin format."),
    ("What is MIDI?", "MIDI is a protocol for musical instrument communication."),
]

# === DEBUG EXAMPLES (less frequent) ===
DEBUG_ISSUES = ["crackling", "stuttering", "glitching", "popping", "buzzing", "distorting"]
DEBUG_CAUSES = ["allocation", "mutex lock", "file I/O", "vector resize", "new operator"]

def gen_greeting():
    """Generate a casual greeting exchange."""
    g = random.choice(GREETINGS)
    r = random.choice(GREETING_RESPONSES)
    return f"<|im_start|>user\n{g}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>\n"

def gen_simple_qa():
    """Generate simple Q&A."""
    q, a = random.choice(SIMPLE_QA)
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"

def gen_debug():
    """Generate debug scenario - ONLY when user mentions plugin issues."""
    issue = random.choice(DEBUG_ISSUES)
    cause = random.choice(DEBUG_CAUSES)
    return f"""<|im_start|>user
My audio plugin is {issue}. Can you help debug it?<|im_end|>
<|im_start|>assistant
<thought>User has audio issue: {issue}. Check for realtime violations.</thought>
Likely cause: {cause} in audio thread. Move heavy operations to prepareToPlay().<|im_end|>
"""

def gen_code():
    """Generate code request."""
    classes = ["Delay", "Reverb", "Compressor", "EQ", "Limiter", "Filter"]
    cls = random.choice(classes)
    return f"""<|im_start|>user
Write a simple {cls} class<|im_end|>
<|im_start|>assistant
<thought>User wants a {cls} class. Keep it minimal and complete.</thought>
```cpp
class {cls} {{
    float param = 0.5f;
public:
    void process(float* data, int n) {{
        for (int i = 0; i < n; ++i) data[i] *= param;
    }}
}};
```<|im_end|>
"""

def gen_fim():
    """Generate fill-in-middle example."""
    cls = random.choice(["Delay", "Reverb", "Filter", "Gain"])
    val = round(random.uniform(0.1, 1.0), 2)
    return f"<PRE>void {cls}::process(float* d, int n) {{\n    <SUF>\n}}<MID>for(int i=0;i<n;++i)d[i]*={val}f;"

def generate_corpus(filepath, entries=10000):
    print(f"Generating {entries} balanced examples...")
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(entries):
            r = random.random()
            # 40% Greetings/Simple Chat - MOST COMMON
            if r < 0.40:
                f.write(gen_greeting() if random.random() < 0.6 else gen_simple_qa())
            # 25% Debug - ONLY when explicitly asking about issues
            elif r < 0.65:
                f.write(gen_debug())
            # 20% Code requests
            elif r < 0.85:
                f.write(gen_code())
            # 15% FIM
            else:
                f.write(gen_fim())
            
            if i % 2000 == 0:
                print(f"  {i}/{entries}...")
    print("Done.")

if __name__ == "__main__":
    os.makedirs("c:/vernex/data", exist_ok=True)
    generate_corpus("c:/vernex/data/audio_corpus.txt", entries=10000)
