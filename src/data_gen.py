"""
Code-focused training data for Vernex 500M.
Priority: Code generation, then search for unknowns, minimal chat.
"""
import random
import os

# === MINIMAL CHAT (just enough to be polite) ===
GREETINGS = ["Hi", "Hello", "Hey", "Hi there"]
GREETING_RESPONSES = ["Hi! What code do you need?", "Hello! Show me the code.", "Hey! What's the task?"]

# === CODE TEMPLATES ===
CLASSES = ["Delay", "Reverb", "Compressor", "EQ", "Limiter", "Filter", "Phaser", "Chorus", "Distortion", "Gate", "Saturator", "Tremolo"]
PARAMS = ["gain", "mix", "feedback", "time", "rate", "depth", "threshold", "ratio", "attack", "release"]

def gen_greeting():
    return f"<|im_start|>user\n{random.choice(GREETINGS)}<|im_end|>\n<|im_start|>assistant\n{random.choice(GREETING_RESPONSES)}<|im_end|>\n"

def gen_simple_class():
    cls = random.choice(CLASSES)
    param = random.choice(PARAMS)
    val = round(random.uniform(0.1, 1.0), 2)
    return f"""<|im_start|>user
Write a {cls} class<|im_end|>
<|im_start|>assistant
```cpp
class {cls} {{
    float {param} = {val}f;
public:
    void process(float* data, int n) {{
        for (int i = 0; i < n; ++i) {{
            data[i] *= {param};
        }}
    }}
    void set{param.capitalize()}(float v) {{ {param} = v; }}
}};
```<|im_end|>
"""

def gen_full_plugin():
    cls = random.choice(CLASSES)
    return f"""<|im_start|>user
Write a complete JUCE {cls} plugin processor<|im_end|>
<|im_start|>assistant
```cpp
#include <JuceHeader.h>

class {cls}Processor : public juce::AudioProcessor {{
public:
    {cls}Processor() : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo())
        .withOutput("Output", juce::AudioChannelSet::stereo())) {{}}
    
    void prepareToPlay(double sr, int bs) override {{ sampleRate = sr; }}
    void releaseResources() override {{}}
    
    void processBlock(juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override {{
        for (int ch = 0; ch < buf.getNumChannels(); ++ch) {{
            float* data = buf.getWritePointer(ch);
            for (int i = 0; i < buf.getNumSamples(); ++i) {{
                data[i] *= gain;
            }}
        }}
    }}
    
    const juce::String getName() const override {{ return "{cls}"; }}
    bool acceptsMidi() const override {{ return false; }}
    bool producesMidi() const override {{ return false; }}
    double getTailLengthSeconds() const override {{ return 0.0; }}
    
    int getNumPrograms() override {{ return 1; }}
    int getCurrentProgram() override {{ return 0; }}
    void setCurrentProgram(int) override {{}}
    const juce::String getProgramName(int) override {{ return {{}}; }}
    void changeProgramName(int, const juce::String&) override {{}}
    
    void getStateInformation(juce::MemoryBlock&) override {{}}
    void setStateInformation(const void*, int) override {{}}
    
    juce::AudioProcessorEditor* createEditor() override {{ return nullptr; }}
    bool hasEditor() const override {{ return false; }}

private:
    double sampleRate = 44100.0;
    float gain = 1.0f;
}};
```<|im_end|>
"""

def gen_dsp_function():
    funcs = [
        ("lowpass filter", "y[i] = y[i-1] + alpha * (x[i] - y[i-1]);"),
        ("highpass filter", "y[i] = alpha * (y[i-1] + x[i] - x[i-1]);"),
        ("soft clipper", "y[i] = tanh(x[i] * drive) / tanh(drive);"),
        ("hard clipper", "y[i] = std::clamp(x[i], -threshold, threshold);"),
        ("ring modulator", "y[i] = x[i] * sin(2.0 * M_PI * freq * i / sr);"),
    ]
    name, code = random.choice(funcs)
    return f"""<|im_start|>user
Implement a {name}<|im_end|>
<|im_start|>assistant
```cpp
void process(float* x, float* y, int n) {{
    for (int i = 0; i < n; ++i) {{
        {code}
    }}
}}
```<|im_end|>
"""

def gen_search_example():
    topics = ["JUCE 8 new features", "real-time audio best practices", "lock-free queue implementation", "SIMD optimization for audio"]
    topic = random.choice(topics)
    return f"""<|im_start|>user
What are the {topic}?<|im_end|>
<|im_start|>assistant
<thought>I don't have specific info on this. Let me search.</thought>
[SEARCH: {topic}]<|im_end|>
"""

def gen_debug_with_search():
    issues = ["crackling", "latency spikes", "memory leak", "CPU overload"]
    issue = random.choice(issues)
    return f"""<|im_start|>user
My plugin has {issue}, what could cause it?<|im_end|>
<|im_start|>assistant
<thought>Audio {issue}. Check common causes then search if needed.</thought>
Common causes of {issue}:
1. Allocation in processBlock()
2. Blocking calls (mutex, file I/O)
3. Unbounded buffer sizes

[SEARCH: JUCE plugin {issue} causes]<|im_end|>
"""

def gen_fim():
    cls = random.choice(CLASSES)
    param = random.choice(PARAMS)
    val = round(random.uniform(0.1, 1.0), 2)
    return f"<PRE>void {cls}::process(float* d, int n) {{\n    <SUF>\n}}<MID>for(int i=0;i<n;++i)d[i]*={val}f;"

def gen_uncertainty():
    return """<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
I focus on audio/code. For general knowledge, try a search:
[SEARCH: capital of France]<|im_end|>
"""

def generate_corpus(filepath, entries=15000):
    print(f"Generating {entries} code-focused examples...")
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(entries):
            r = random.random()
            # 60% Code (main focus)
            if r < 0.25:
                f.write(gen_simple_class())
            elif r < 0.40:
                f.write(gen_full_plugin())
            elif r < 0.60:
                f.write(gen_dsp_function())
            # 15% FIM
            elif r < 0.75:
                f.write(gen_fim())
            # 15% Debug + Search
            elif r < 0.85:
                f.write(gen_debug_with_search() if random.random() < 0.5 else gen_search_example())
            # 10% Chat + Uncertainty
            else:
                f.write(gen_greeting() if random.random() < 0.5 else gen_uncertainty())
            
            if i % 3000 == 0:
                print(f"  {i}/{entries}...")
    print("Done.")

if __name__ == "__main__":
    os.makedirs("c:/vernex/data", exist_ok=True)
    generate_corpus("c:/vernex/data/audio_corpus.txt", entries=15000)
