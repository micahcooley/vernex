
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# Templates for synthetic tool use
SCENARIOS = [
    {
        "user": "There's a bug in src/audio_processor.cpp where the buffer isn't clearing.",
        "file": "src/audio_processor.cpp",
        "thought": "I need to read the file first to understand the context.",
        "tool_call": '{"tool": "read_file", "path": "src/audio_processor.cpp"}',
        "tool_output": "class AudioProcessor { void process(AudioBuffer& buffer) { // Missing clear } };",
        "next_thought": "I see the issue. The buffer needs to be cleared before processing.",
        "next_tool_call": '{"tool": "edit_file", "path": "src/audio_processor.cpp", "find": "// Missing clear", "replace": "buffer.clear();"}'
    },
    {
        "user": "Add a new maximize button to the main window.",
        "file": "src/main_window.cpp",
        "thought": "I should search for where the window is initialized.",
        "tool_call": '{"tool": "grep_search", "query": "MainWindow", "path": "src"}',
        "tool_output": "src/main_window.cpp:10: MainWindow::MainWindow() : DocumentWindow(...) { ... }",
        "next_thought": "Okay, I'll add the resizable flag to the constructor.",
        "next_tool_call": '{"tool": "edit_file", "path": "src/main_window.cpp", "find": "DocumentWindow(...)", "replace": "DocumentWindow(..., DocumentWindow::allButtons)"}'
    },
    {
        "user": "Fix the compile error about 'SkPaint' not being found.",
        "file": "src/component.cpp",
        "thought": "It looks like a missing include. I'll read the file imports.",
        "tool_call": '{"tool": "read_file", "path": "src/component.cpp", "lines": "1-20"}',
        "tool_output": "#include <juce_core/juce_core.h>\n// Missing Skia includes",
        "next_thought": "I need to include Skia headers.",
        "next_tool_call": '{"tool": "edit_file", "path": "src/component.cpp", "find": "#include <juce_core/juce_core.h>", "replace": "#include <juce_core/juce_core.h>\n#include <include/core/SkPaint.h>"}'
    },
    {
        "user": "Implement a gain function in the processBlock.",
        "file": "src/plugin_processor.cpp",
        "thought": "I'll create a simple gain loop.",
        "tool_call": '{"tool": "edit_file", "path": "src/plugin_processor.cpp", "find": "// gain logic", "replace": "for (int i=0; i<samples; ++i) buffer[i] *= gain;"}',
        "tool_output": "File updated successfully.",
        "next_thought": "I should also check if the user wants smoothed gain.",
        "next_tool_call": "CRITIC: Wait, is 'gain' smoothed? Won't this cause audio clicks? Why use a raw loop instead of SIMD or JUCE FloatVectorOperations?"
    },
    {
        "user": "Create an instance of my custom Component.",
        "file": "src/main.cpp",
        "thought": "I will add it as a member and new it in constructor.",
        "tool_call": '{"tool": "edit_file", "path": "src/main.cpp", "find": "public:", "replace": "public: std::unique_ptr<MyComp> comp;"}',
        "tool_output": "Updated header.",
        "next_thought": "Now I will initialize it.",
        "next_tool_call": "CRITIC: Why unique_ptr? Is this component small enough to be a direct member? Also, where is the visibility set? Are you sure this won't crash if the parent is deleted first? This looks like a 2010 way of doing things."
    },
    {
        "user": "Design a basic volume slider for the UI.",
        "file": "src/gui.cpp",
        "thought": "I'll use a standard JUCE Slider.",
        "tool_call": '{"tool": "edit_file", "path": "src/gui.cpp", "find": "// slider init", "replace": "slider.setSliderStyle(Slider::LinearHorizontal);\naddAndMakeVisible(slider);"}',
        "tool_output": "Slider added.",
        "next_thought": "I'll set the bounds now.",
        "next_tool_call": "CRITIC: Yea but users will expect a custom LookAndFeel here. This default JUCE slider looks ugly and dated. Can we use a Skia-drawn custom knob instead?"
    },
    {
        "user": "How do I use the web search tool?",
        "file": "N/A",
        "thought": "The user is asking about my capabilities. I should explain or demonstrate.",
        "tool_call": '{"tool": "web_search", "query": "latest Juce documentation"}',
        "tool_output": "JUCE 8.0 Released: New features include...",
        "next_thought": "I have the info. I will summarize it for the user.",
        "next_tool_call": "FINAL_RESPONSE: I can search the web! For example, JUCE 8.0 just released..."
    }
]

def generate_tool_corpus(output_path, num_samples=2000):
    print(f"Generating {num_samples} tool-use samples...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            scenario = random.choice(SCENARIOS)
            
            # vary the prompt slightly
            prompt_variations = [
                scenario["user"],
                f"Can you help me with this? {scenario['user']}",
                f"IMPORTANT: {scenario['user']}",
                f"Debug this: {scenario['user']}"
            ]
            prompt = random.choice(prompt_variations)
            
            # Format: <|user|> ... <|model|> <|thought|> ... <|tool_call|> ...
            # We want the model to learn to predict the thought AND the tool call
            
            # Step 1: User -> Thought -> Tool Call
            sample = f"<|user|>\n{prompt}\n<|model|>\n"
            sample += f"<|thought|>\n{scenario['thought']}\n"
            sample += f"<|tool_call|>\n{scenario['tool_call']}\n"
            sample += f"<|tool_output|>\n{scenario['tool_output']}\n"
            
            # Step 2: Tool Output -> Next Thought -> Next Tool Call (or verification)
            sample += f"<|thought|>\n{scenario['next_thought']}\n"
            if "FINAL_RESPONSE" in scenario['next_tool_call']:
                 sample += f"{scenario['next_tool_call']}\n"
            else:
                 sample += f"<|tool_call|>\n{scenario['next_tool_call']}\n"
            
            sample += "<|im_end|>\n"
            f.write(sample)
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    out = DATA_DIR / "tool_corpus.txt"
    generate_tool_corpus(out)
