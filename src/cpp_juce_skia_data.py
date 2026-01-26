"""
Enhanced training data for Vernex 500M - C++/JUCE/Skia focus.
70% Code (JUCE, Skia, DSP), 30% Debugging/Chat/FIM.
No sugarcoating, respectful roasting style.
"""
import random

# === JUCE PATTERNS ===

JUCE_AUDIO_PROCESSOR = '''<|im_start|>user
Write a JUCE AudioProcessor for a simple gain plugin<|im_end|>
<|im_start|>assistant
<thought>User wants a basic gain plugin. Keep it minimal, no fluff.</thought>
```cpp
class GainProcessor : public juce::AudioProcessor {
public:
    GainProcessor() : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo())
        .withOutput("Output", juce::AudioChannelSet::stereo())) {}

    void prepareToPlay(double, int) override {}
    void releaseResources() override {}

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override {
        float gain = *apvts.getRawParameterValue("gain");
        buffer.applyGain(gain);
    }

    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }
    const juce::String getName() const override { return "Gain"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    juce::AudioProcessorValueTreeState apvts{*this, nullptr, "Params",
        {std::make_unique<juce::AudioParameterFloat>("gain", "Gain", 0.0f, 2.0f, 1.0f)}};
};
```
That's it. No unnecessary abstractions.<|im_end|>
'''

JUCE_BUFFER_OPERATIONS = '''<|im_start|>user
How do I copy one AudioBuffer channel to another?<|im_end|>
<|im_start|>assistant
```cpp
// Copy channel 0 to channel 1
buffer.copyFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());

// Or if you want to add instead of replace:
buffer.addFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());
```
Don't loop through samples manually like a beginner. Use the built-in methods.<|im_end|>
'''

JUCE_MIDI_PROCESSING = '''<|im_start|>user
How do I process MIDI note on/off in JUCE?<|im_end|>
<|im_start|>assistant
```cpp
void processBlock(juce::AudioBuffer<float>& audio, juce::MidiBuffer& midi) {
    for (const auto metadata : midi) {
        auto msg = metadata.getMessage();
        int sample = metadata.samplePosition;
        
        if (msg.isNoteOn()) {
            int note = msg.getNoteNumber();
            float velocity = msg.getFloatVelocity();
            // Trigger your synth voice here
        }
        else if (msg.isNoteOff()) {
            int note = msg.getNoteNumber();
            // Release your synth voice here
        }
    }
}
```
The `metadata.samplePosition` is crucial for sample-accurate timing. Ignore it and your timing will suck.<|im_end|>
'''

JUCE_THREAD_SAFETY = '''<|im_start|>user
Is it safe to call repaint() from the audio thread?<|im_end|>
<|im_start|>assistant
No. Absolutely not. That's a recipe for deadlocks.

The audio thread must NEVER block. `repaint()` can trigger UI operations that lock mutexes.

Use an atomic flag:
```cpp
// In your processor:
std::atomic<bool> shouldRepaint{false};

void processBlock(...) {
    // ... audio processing ...
    shouldRepaint.store(true, std::memory_order_release);
}

// In your editor, poll with a timer:
void timerCallback() override {
    if (processor.shouldRepaint.exchange(false, std::memory_order_acquire)) {
        levelMeter.repaint();
    }
}
```
Lock-free or go home.<|im_end|>
'''

JUCE_APVTS = '''<|im_start|>user
How do I properly set up APVTS parameters?<|im_end|>
<|im_start|>assistant
```cpp
juce::AudioProcessorValueTreeState::ParameterLayout createParams() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // Float parameter with range
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"attack", 1},  // ID, version hint
        "Attack",                         // Name
        juce::NormalisableRange<float>(0.001f, 2.0f, 0.001f, 0.5f), // range, step, skew
        0.01f                             // Default
    ));
    
    // Choice parameter
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{"mode", 1},
        "Mode",
        juce::StringArray{"Clean", "Warm", "Aggressive"},
        0  // Default index
    ));
    
    return {params.begin(), params.end()};
}
```
The version hint in ParameterID is for DAW recall compatibility. Don't skip it.<|im_end|>
'''

# === SKIA PATTERNS ===

SKIA_BASIC_DRAWING = '''<|im_start|>user
How do I draw a rounded rectangle with Skia?<|im_end|>
<|im_start|>assistant
```cpp
void draw(SkCanvas* canvas) {
    SkPaint paint;
    paint.setAntiAlias(true);
    paint.setColor(SkColorSetARGB(255, 30, 30, 40));
    paint.setStyle(SkPaint::kFill_Style);
    
    SkRect rect = SkRect::MakeXYWH(10, 10, 200, 100);
    float radius = 8.0f;
    
    canvas->drawRoundRect(rect, radius, radius, paint);
    
    // Add a border
    paint.setStyle(SkPaint::kStroke_Style);
    paint.setStrokeWidth(1.0f);
    paint.setColor(SkColorSetARGB(100, 255, 255, 255));
    canvas->drawRoundRect(rect, radius, radius, paint);
}
```
Always enable anti-aliasing unless you WANT the jaggies.<|im_end|>
'''

SKIA_GRADIENT = '''<|im_start|>user
How do I create a linear gradient in Skia?<|im_end|>
<|im_start|>assistant
```cpp
SkPaint paint;
paint.setAntiAlias(true);

SkPoint points[2] = {
    SkPoint::Make(0, 0),
    SkPoint::Make(0, 100)  // Vertical gradient
};
SkColor colors[2] = {
    SkColorSetARGB(255, 60, 60, 80),   // Top
    SkColorSetARGB(255, 20, 20, 30)    // Bottom
};

paint.setShader(SkGradientShader::MakeLinear(
    points, colors, nullptr, 2, SkTileMode::kClamp));

canvas->drawRect(SkRect::MakeWH(200, 100), paint);
```
The nullptr is for color positions - pass nullptr for even distribution.<|im_end|>
'''

SKIA_TEXT = '''<|im_start|>user
How do I render text with Skia?<|im_end|>
<|im_start|>assistant
```cpp
void drawText(SkCanvas* canvas, const char* text, float x, float y) {
    SkFont font;
    font.setSize(14.0f);
    font.setEmbolden(false);
    font.setEdging(SkFont::Edging::kSubpixelAntiAlias);
    
    SkPaint paint;
    paint.setAntiAlias(true);
    paint.setColor(SK_ColorWHITE);
    
    canvas->drawSimpleText(text, strlen(text), SkTextEncoding::kUTF8, x, y, font, paint);
}

// For proper font loading:
sk_sp<SkTypeface> typeface = SkTypeface::MakeFromFile("/path/to/font.ttf");
if (typeface) {
    font.setTypeface(typeface);
}
```
Use SubpixelAntiAlias for clean text. Regular antialiasing looks blurry on LCD screens.<|im_end|>
'''

SKIA_CLIP = '''<|im_start|>user
How do I clip drawing to a region in Skia?<|im_end|>
<|im_start|>assistant
```cpp
void drawClipped(SkCanvas* canvas) {
    canvas->save();  // ALWAYS save before clipping
    
    SkRect clipRect = SkRect::MakeXYWH(50, 50, 200, 100);
    canvas->clipRect(clipRect);
    
    // Everything drawn here is clipped to clipRect
    SkPaint paint;
    paint.setColor(SK_ColorRED);
    canvas->drawCircle(100, 100, 80, paint);  // Will be clipped
    
    canvas->restore();  // Restore removes the clip
    
    // Drawing here is not clipped
    paint.setColor(SK_ColorBLUE);
    canvas->drawCircle(250, 100, 30, paint);
}
```
Forgetting `save()`/`restore()` is a classic bug. The clip will leak into other draw calls.<|im_end|>
'''

SKIA_GPU_CONTEXT = '''<|im_start|>user
How do I set up Skia with GPU rendering?<|im_end|>
<|im_start|>assistant
```cpp
// 1. Create GL context (platform-specific, use your windowing system)
// 2. Get Skia's GL interface
sk_sp<const GrGLInterface> interface = GrGLMakeNativeInterface();
if (!interface) {
    DBG("Failed to get GL interface");
    return;
}

// 3. Create GrDirectContext
GrContextOptions options;
options.fAllowPathMaskCaching = true;
sk_sp<GrDirectContext> grContext = GrDirectContext::MakeGL(interface, options);

// 4. Create surface from framebuffer
GrGLFramebufferInfo fbInfo;
fbInfo.fFBOID = 0;  // Default framebuffer
fbInfo.fFormat = GL_RGBA8;

GrBackendRenderTarget target(width, height, 0, 8, fbInfo);
sk_sp<SkSurface> surface = SkSurfaces::WrapBackendRenderTarget(
    grContext.get(), target, kBottomLeft_GrSurfaceOrigin,
    kRGBA_8888_SkColorType, nullptr, nullptr);

SkCanvas* canvas = surface->getCanvas();
// Draw with GPU acceleration
```
Check `grContext->abandoned()` every frame on Wayland. The context can die at any time.<|im_end|>
'''

# === DSP PATTERNS ===

DSP_BIQUAD = '''<|im_start|>user
Write a biquad filter in C++<|im_end|>
<|im_start|>assistant
```cpp
class Biquad {
public:
    void setCoefficients(float b0, float b1, float b2, float a1, float a2) {
        b0_ = b0; b1_ = b1; b2_ = b2;
        a1_ = a1; a2_ = a2;
    }
    
    float process(float input) {
        float output = b0_ * input + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
        x2_ = x1_; x1_ = input;
        y2_ = y1_; y1_ = output;
        return output;
    }
    
    void reset() { x1_ = x2_ = y1_ = y2_ = 0.0f; }
    
    // Lowpass design
    void setLowpass(float freq, float q, float sampleRate) {
        float w0 = 2.0f * M_PI * freq / sampleRate;
        float cosw0 = std::cos(w0);
        float sinw0 = std::sin(w0);
        float alpha = sinw0 / (2.0f * q);
        
        float a0 = 1.0f + alpha;
        b0_ = ((1.0f - cosw0) / 2.0f) / a0;
        b1_ = (1.0f - cosw0) / a0;
        b2_ = b0_;
        a1_ = (-2.0f * cosw0) / a0;
        a2_ = (1.0f - alpha) / a0;
    }

private:
    float b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
    float x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
};
```
This is Direct Form I. For modulating cutoff, use Direct Form II Transposed to avoid clicks.<|im_end|>
'''

DSP_DELAY = '''<|im_start|>user
Implement a basic delay line<|im_end|>
<|im_start|>assistant
```cpp
class DelayLine {
public:
    void setMaxDelay(int maxSamples) {
        buffer_.resize(maxSamples, 0.0f);
        writePos_ = 0;
    }
    
    void setDelay(int samples) {
        delaySamples_ = std::min(samples, (int)buffer_.size() - 1);
    }
    
    float process(float input) {
        // Write
        buffer_[writePos_] = input;
        
        // Read with wrap
        int readPos = writePos_ - delaySamples_;
        if (readPos < 0) readPos += buffer_.size();
        
        float output = buffer_[readPos];
        
        // Advance write position
        writePos_++;
        if (writePos_ >= buffer_.size()) writePos_ = 0;
        
        return output;
    }
    
    // For fractional delay (interpolated)
    float processInterpolated(float input, float delaySamples) {
        buffer_[writePos_] = input;
        
        float frac = delaySamples - std::floor(delaySamples);
        int idx0 = writePos_ - (int)delaySamples;
        int idx1 = idx0 - 1;
        if (idx0 < 0) idx0 += buffer_.size();
        if (idx1 < 0) idx1 += buffer_.size();
        
        float output = buffer_[idx0] * (1.0f - frac) + buffer_[idx1] * frac;
        
        writePos_ = (writePos_ + 1) % buffer_.size();
        return output;
    }

private:
    std::vector<float> buffer_;
    int writePos_ = 0;
    int delaySamples_ = 0;
};
```
Linear interpolation is fine for delays. Use allpass interpolation if you're a perfectionist.<|im_end|>
'''

DSP_ENVELOPE = '''<|im_start|>user
Write an ADSR envelope<|im_end|>
<|im_start|>assistant
```cpp
class ADSR {
public:
    enum State { Idle, Attack, Decay, Sustain, Release };
    
    void setSampleRate(float sr) { sampleRate_ = sr; updateRates(); }
    void setAttack(float ms) { attackMs_ = ms; updateRates(); }
    void setDecay(float ms) { decayMs_ = ms; updateRates(); }
    void setSustain(float level) { sustainLevel_ = level; }
    void setRelease(float ms) { releaseMs_ = ms; updateRates(); }
    
    void noteOn() { state_ = Attack; }
    void noteOff() { if (state_ != Idle) state_ = Release; }
    
    float process() {
        switch (state_) {
            case Attack:
                value_ += attackRate_;
                if (value_ >= 1.0f) { value_ = 1.0f; state_ = Decay; }
                break;
            case Decay:
                value_ -= decayRate_;
                if (value_ <= sustainLevel_) { value_ = sustainLevel_; state_ = Sustain; }
                break;
            case Sustain:
                value_ = sustainLevel_;
                break;
            case Release:
                value_ -= releaseRate_;
                if (value_ <= 0.0f) { value_ = 0.0f; state_ = Idle; }
                break;
            case Idle:
                value_ = 0.0f;
                break;
        }
        return value_;
    }
    
    bool isActive() const { return state_ != Idle; }

private:
    void updateRates() {
        attackRate_ = 1.0f / (attackMs_ * 0.001f * sampleRate_);
        decayRate_ = (1.0f - sustainLevel_) / (decayMs_ * 0.001f * sampleRate_);
        releaseRate_ = sustainLevel_ / (releaseMs_ * 0.001f * sampleRate_);
    }
    
    State state_ = Idle;
    float value_ = 0.0f;
    float sampleRate_ = 44100.0f;
    float attackMs_ = 10.0f, decayMs_ = 100.0f, sustainLevel_ = 0.7f, releaseMs_ = 200.0f;
    float attackRate_ = 0.0f, decayRate_ = 0.0f, releaseRate_ = 0.0f;
};
```
This is linear. For musical envelopes, use exponential curves: `value_ *= decayCoeff_`.<|im_end|>
'''

# === DEBUGGING SCENARIOS ===

DEBUG_MEMORY_LEAK = '''<|im_start|>user
My plugin is leaking memory, how do I find it?<|im_end|>
<|im_start|>assistant
1. **Enable JUCE leak detector** - already on by default in debug builds. Check output for "JUCE MEMORY LEAK" on shutdown.

2. **AddressSanitizer** (better):
```bash
# In CMakeLists.txt
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(YourPlugin PRIVATE -fsanitize=address)
    target_link_options(YourPlugin PRIVATE -fsanitize=address)
endif()
```

3. **Common culprits in JUCE plugins:**
```cpp
// BAD - Listener not removed
class MyComponent : public Button::Listener {
    MyComponent() { button.addListener(this); }
    // Destructor never removes listener = CRASH or LEAK
};

// GOOD
~MyComponent() { button.removeListener(this); }
```

4. **Check your Timer callbacks:**
```cpp
~MyComponent() {
    stopTimer();  // REQUIRED or it can fire after destruction
}
```

5. **std::unique_ptr for owned objects:**
```cpp
// BAD
Processor* proc = new Processor();  // Who deletes this?

// GOOD
std::unique_ptr<Processor> proc = std::make_unique<Processor>();
```

Run `valgrind --leak-check=full ./YourPlugin` as the nuclear option.<|im_end|>
'''

DEBUG_CRACKLE = '''<|im_start|>user
My audio has crackling/popping, what's wrong?<|im_end|>
<|im_start|>assistant
Ordered by likelihood:

1. **Buffer underrun** - Your processBlock is too slow
   - Profile with a timer: `auto start = std::chrono::high_resolution_clock::now();`
   - If processing takes > buffer_size/sample_rate seconds, you're screwed

2. **Denormals** - Tiny floating point numbers tank CPU
```cpp
// Add at start of processBlock:
juce::ScopedNoDenormals noDenormals;
```

3. **Discontinuities** - Parameter jumps cause clicks
```cpp
// BAD
float gain = *params.getRawParameterValue("gain");
buffer.applyGain(gain);  // Sudden jump = click

// GOOD
float targetGain = *params.getRawParameterValue("gain");
for (int i = 0; i < numSamples; ++i) {
    smoothedGain.setTargetValue(targetGain);
    float g = smoothedGain.getNextValue();
    sample[i] *= g;
}
```

4. **Blocking on audio thread** - Memory allocation, locks, file I/O
```cpp
// NEVER in processBlock:
new / delete / malloc / free
std::mutex::lock()
File operations
```

5. **Wrong buffer handling** - Are you reading past buffer end?
```cpp
for (int i = 0; i < buffer.getNumSamples(); ++i)  // GOOD
for (int i = 0; i <= buffer.getNumSamples(); ++i) // BAD - off by one
```<|im_end|>
'''

DEBUG_THREADING = '''<|im_start|>user
I'm getting random crashes, sometimes works sometimes doesn't<|im_end|>
<|im_start|>assistant
Random crashes = race condition. Classic.

**Step 1: Identify the threads**
- Audio thread: `processBlock()`, high priority, NEVER block
- Message thread: UI, `paint()`, `resized()`, timers
- Other: background loading, analysis

**Step 2: Check shared state access**
```cpp
// BAD - both threads read/write
class Processor {
    float gain = 1.0f;  // Audio reads, UI writes = race
};

// GOOD - atomic for simple values
std::atomic<float> gain{1.0f};

// For complex data, use lock-free queue
juce::AbstractFifo fifo(1024);
```

**Step 3: ThreadSanitizer**
```bash
target_compile_options(YourPlugin PRIVATE -fsanitize=thread)
target_link_options(YourPlugin PRIVATE -fsanitize=thread)
```
This WILL find your race conditions.

**Step 4: Common patterns**
```cpp
// Audio -> UI communication
std::atomic<float> levelForMeter;  // Audio writes, timer polls

// UI -> Audio communication  
std::atomic<bool> paramChanged;  // UI sets, audio checks and clears

// Complex data (presets, wavetables)
// Load in background, swap pointer atomically
std::atomic<WaveTable*> currentTable;
```

Never use `juce::CriticalSection` on audio thread. It blocks.<|im_end|>
'''

# === CRITIC/VERIFICATION EXAMPLES ===

CRITIC_WRONG_API = '''<|im_start|>system
You are watching Agent-1's response. Critique any errors.<|im_end|>
<|im_start|>user
How do I get the sample rate in JUCE?<|im_end|>
<|im_start|>agent-1
Use `juce::AudioDeviceManager::getSampleRate()` to get the current sample rate.<|im_end|>
<|im_start|>critic
That's wrong. In an AudioProcessor, you get it from `getSampleRate()` which is a member function, called AFTER `prepareToPlay()`. The AudioDeviceManager approach is for standalone apps, not plugins. Correct answer:
```cpp
void prepareToPlay(double sampleRate, int samplesPerBlock) override {
    currentSampleRate_ = sampleRate;  // Store it here
}
```
Or just call `getSampleRate()` anywhere in your processor after prepare.<|im_end|>
'''

CRITIC_OUTDATED = '''<|im_start|>system
You are watching Agent-1's response. Verify with web search if needed.<|im_end|>
<|im_start|>user
What's the latest JUCE version?<|im_end|>
<|im_start|>agent-1
JUCE 7.0.5 is the latest version.<|im_end|>
<|im_start|>critic
That's outdated. [SEARCH: latest JUCE version 2024]
After checking, JUCE 8 has been released. Agent-1 is giving stale information. Always verify version numbers with a search.<|im_end|>
'''

CRITIC_CORRECT = '''<|im_start|>system
You are watching Agent-1's response. Verify correctness.<|im_end|>
<|im_start|>user
How do I clear a JUCE AudioBuffer?<|im_end|>
<|im_start|>agent-1
```cpp
buffer.clear();
```
This zeroes all channels and samples in the buffer.<|im_end|>
<|im_start|>critic
Correct. Simple, direct, accurate. Nothing to add.<|im_end|>
'''

CRITIC_UNSAFE = '''<|im_start|>system
You are watching Agent-1's response. Flag any unsafe practices.<|im_end|>
<|im_start|>user
How do I update a UI label from processBlock?<|im_end|>
<|im_start|>agent-1
Easy, just call `myLabel.setText("value")` from inside processBlock.<|im_end|>
<|im_start|>critic
STOP. That's a race condition waiting to happen. setText() allocates memory and accesses the message thread. Calling it from audio thread = undefined behavior, potential deadlock, or crash.

Correct pattern:
```cpp
// In processor
std::atomic<float> valueForUI{0.0f};

void processBlock(...) {
    valueForUI.store(computedValue, std::memory_order_relaxed);
}

// In editor, use a Timer
void timerCallback() override {
    label.setText(String(processor.valueForUI.load()));
}
```<|im_end|>
'''

# === FIM PATTERNS ===

FIM_SAMPLES = [
    "<PRE>void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {\n    <SUF>\n}<MID>for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {\n        auto* data = buffer.getWritePointer(ch);\n        for (int i = 0; i < buffer.getNumSamples(); ++i) {\n            data[i] *= gain_;\n        }\n    }",
    "<PRE>class DelayLine {\n    std::vector<float> buffer_;\n    int writePos_ = 0;\npublic:\n    <SUF>\n};<MID>void setSize(int samples) { buffer_.resize(samples, 0.0f); }\n    float read(int delay) { return buffer_[(writePos_ - delay + buffer_.size()) % buffer_.size()]; }\n    void write(float sample) { buffer_[writePos_] = sample; writePos_ = (writePos_ + 1) % buffer_.size(); }",
    "<PRE>void paint(SkCanvas* canvas) {\n    SkPaint paint;\n    <SUF>\n    canvas->drawRect(rect, paint);\n}<MID>paint.setAntiAlias(true);\n    paint.setColor(SkColorSetARGB(255, 40, 40, 50));\n    SkRect rect = SkRect::MakeWH(100, 50);",
]

# === COLLECTION ===

JUCE_SAMPLES = [JUCE_AUDIO_PROCESSOR, JUCE_BUFFER_OPERATIONS, JUCE_MIDI_PROCESSING, 
                JUCE_THREAD_SAFETY, JUCE_APVTS]
SKIA_SAMPLES = [SKIA_BASIC_DRAWING, SKIA_GRADIENT, SKIA_TEXT, SKIA_CLIP, SKIA_GPU_CONTEXT]
DSP_SAMPLES = [DSP_BIQUAD, DSP_DELAY, DSP_ENVELOPE]
DEBUG_SAMPLES = [DEBUG_MEMORY_LEAK, DEBUG_CRACKLE, DEBUG_THREADING]
CRITIC_SAMPLES = [CRITIC_WRONG_API, CRITIC_OUTDATED, CRITIC_CORRECT, CRITIC_UNSAFE]

def generate_training_data(filepath, entries=20000):
    """
    Generate training corpus with 70/30 split.
    70% = JUCE/Skia/C++ code
    30% = Debugging, FIM, Critic examples, Basic chat
    """
    print(f"Generating {entries} training examples (70% code, 30% other)...")
    
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(entries):
            r = random.random()
            
            # 70% Code (JUCE/Skia/DSP)
            if r < 0.70:
                pool = JUCE_SAMPLES + SKIA_SAMPLES + DSP_SAMPLES
                f.write(random.choice(pool))
            
            # 12% Debugging
            elif r < 0.82:
                f.write(random.choice(DEBUG_SAMPLES))
            
            # 10% FIM
            elif r < 0.92:
                f.write(random.choice(FIM_SAMPLES) + "\n")
            
            # 5% Critic examples (for Agent-2 training)
            elif r < 0.97:
                f.write(random.choice(CRITIC_SAMPLES))
            
            # 3% Basic greetings
            else:
                greetings = ["Hi", "Hello", "Hey"]
                responses = ["What code do you need?", "Show me the problem.", "Let's debug."]
                f.write(f"<|im_start|>user\n{random.choice(greetings)}<|im_end|>\n")
                f.write(f"<|im_start|>assistant\n{random.choice(responses)}<|im_end|>\n")
            
            if i % 5000 == 0 and i > 0:
                print(f"  {i}/{entries}...")
    
    print(f"Done. Saved to {filepath}")

if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)
    generate_training_data(str(DATA_DIR / "cpp_juce_skia_corpus.txt"), entries=20000)
