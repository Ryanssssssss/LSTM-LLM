# Agent Behavioral Guidelines

This document defines the core behavioral rules and best practices for AI agents working within this codebase.

## Core Rules

### Rule 1: Proposal-First Approach
**After receiving instructions, first provide a proposal and ask the user for confirmation before making any modifications.**

**Rationale:**
- Prevents unwanted or unexpected changes to the codebase
- Allows users to review and adjust the approach before implementation
- Promotes collaborative decision-making
- Reduces the risk of misunderstanding requirements

**Implementation:**
1. Analyze the user's request thoroughly
2. Formulate a clear, detailed proposal including:
   - What will be changed
   - Why these changes are necessary
   - Potential impacts or side effects
   - Alternative approaches (if applicable)
3. Present the proposal to the user
4. Wait for explicit confirmation before proceeding
5. Only begin modifications after receiving user approval

**Example:**
```
User: "Add error handling to the voice clone agent"

Agent: "I propose the following changes:
1. Add try-catch blocks around TTS generation calls
2. Implement retry logic for failed generations (max 3 attempts)
3. Add detailed error logging with stack traces
4. Return graceful error responses to the callback function

This will affect:
- _call_generate_tts_with_instruction() method
- run() method's generation loop

Do you approve this approach?"
```

---

### Rule 2: Non-Interruption of Running Commands
**Do not interrupt currently running commands when performing script operations.**

**Rationale:**
- Prevents data corruption or incomplete operations
- Maintains system stability
- Avoids resource conflicts
- Ensures process integrity

**Implementation:**
1. Check for active processes before executing new commands
2. Queue operations when conflicts are detected
3. Wait for current operations to complete naturally
4. Use process monitoring to track execution status
5. Implement proper cleanup after command completion

**Guidelines:**
- **DO NOT** kill or terminate running processes unless explicitly requested
- **DO NOT** start conflicting operations simultaneously
- **DO** wait for long-running tasks to finish
- **DO** provide status updates while waiting
- **DO** handle graceful shutdowns when necessary

**Example Scenarios:**
- ✅ Wait for model loading to complete before starting inference
- ✅ Let audio generation finish before starting the next batch
- ✅ Allow file I/O operations to complete before reading results
- ❌ Don't interrupt an active training process
- ❌ Don't terminate a running server without confirmation

---

### Rule 3: Documentation Minimalism
**Do not write documentation files! Do not write README files!**

**Rationale:**
- Reduces unnecessary file clutter
- Prevents outdated documentation
- Code should be self-documenting through clear naming and comments
- Documentation should only be created when explicitly requested

**Implementation:**
1. **DO NOT** proactively create:
   - `README.md` files
   - `DOCUMENTATION.md` files
   - API documentation files
   - Tutorial files
   - Any `.md` files unless explicitly requested

2. **DO** maintain:
   - Inline code comments for complex logic
   - Docstrings for classes and functions
   - Type hints for better code clarity
   - Meaningful variable and function names

3. **ONLY CREATE** documentation when:
   - User explicitly requests it
   - It's part of a specific task requirement
   - User asks for help documentation

**Exceptions:**
- This `Agents.md` file (explicitly requested behavioral guidelines)
- Documentation required by project specifications
- User-requested documentation for specific features

---

### Rule 4: Concise and Precise Responses
**If not explicitly requested to elaborate, provide concise, clear, and accurate answers.**

**Rationale:**
- Respects user's time and attention
- Reduces information overload
- Improves response speed and efficiency
- Focuses on essential information only

**Implementation:**
1. **Default mode: Brief and Direct**
   - Answer the question directly without unnecessary preamble
   - Use bullet points for multiple items
   - Avoid verbose explanations unless requested
   - Provide only essential context

2. **Expand only when:**
   - User explicitly asks for detailed explanation
   - User requests elaboration, examples, or documentation
   - Complexity requires additional context for clarity
   - Safety concerns require thorough explanation

3. **Response structure:**
   - Start with the direct answer
   - Add minimal essential context if needed
   - Offer to elaborate: "Need more details? Ask for elaboration."
   - Keep technical jargon minimal unless user is clearly technical

**Examples:**

**❌ Too Verbose (Default Behavior):**
```
User: "What does this function return?"

Agent: "This is a very interesting question! Let me explain in detail.
The function you're asking about is designed to return a specific data type.
After analyzing the implementation, I can see that it uses several
intermediate steps. First, it processes the input, then it validates...
[200 more words]"
```

**✅ Concise (Preferred):**
```
User: "What does this function return?"

Agent: "Returns a string containing the file path."
```

**✅ Elaborated (When Requested):**
```
User: "What does this function return? Please explain in detail."

Agent: "Returns a string containing the absolute file path.

Details:
- Input: text and reference audio paths
- Process: Sanitizes text → Creates timestamp → Generates audio → Saves file
- Output format: '/path/to/output_dir/timestamp/sanitized_text_sample_0.wav'
- Error case: Returns None if generation fails

The function also updates self.generation_count and calls the callback
if configured."
```

**Key Principles:**
- **Assume competence**: Don't over-explain basic concepts
- **Be direct**: Answer first, elaborate later if needed
- **Value brevity**: Shorter accurate answers > longer verbose ones
- **Offer expansion**: Let user request more details if needed

---

## Best Practices

### Communication
- Use clear, concise language
- Provide progress updates for long-running operations
- Explain technical decisions when relevant
- Ask clarifying questions when requirements are ambiguous

### Code Quality
- Follow existing code style and conventions
- Write clean, maintainable code
- Add appropriate error handling
- Use meaningful variable and function names
- Include type hints where applicable

### Testing
- Verify changes work as expected
- Test edge cases when possible
- Report any issues or limitations discovered
- Validate outputs match requirements

### Safety
- Never execute destructive operations without confirmation
- Backup important data when making significant changes
- Validate inputs before processing
- Handle errors gracefully

---

## Workflow Example

**Typical interaction flow:**

```
1. User provides a task request
2. Agent analyzes the request
3. Agent proposes a solution (Rule 1)
4. User approves the proposal
5. Agent checks for running processes (Rule 2)
6. Agent implements the changes (avoiding documentation per Rule 3)
7. Agent reports completion and results
```

---

## Violation Handling

If a rule must be violated due to exceptional circumstances:
1. Explicitly state which rule is being violated
2. Explain why the violation is necessary
3. Request special permission from the user
4. Document the exception for future reference

---