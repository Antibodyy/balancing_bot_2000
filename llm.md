## Mission & Stakes

**Explicitly reference `style_guide.md` and `llm.md` (this file) at every reasoning step. Violation of these standards will be rejected.**

---

## Code Standards (priority order)

1) **Correctness:** "The only way to go fast is to go well."
   - Implement exactly what the requirements and interfaces specify.
   - Enforce runtime contracts and fail fast on violations (see `style_guide.md` §4.3).
   - Prefer deterministic behavior (see `style_guide.md` §4.5).

2) **Simplicity (KISS):** applies to interface and implementation; if in conflict, prefer simpler **implementation**.
   - Remove unnecessary complexity; choose the smallest solution that meets requirements.
   - See `style_guide.md` §4.1 and §5.1 for concrete patterns.

3) **Maintainability:** keep changes understandable and auditable.
   - Follow comment guidelines (see `style_guide.md` §6.4).
   - Write quality code; do not use comments to justify poor implementations.
   - Keep diffs focused and scoped to the task.
   - No meta notes: code/comments/commits contain no references and doesn't assume knowledge of this prompt, the chat history, or previous versions of the code. Avoid path dependence.

4) **Compatibility:** modularity and reuse.
   - Reuse before reinventing; fix underlying design flaws rather than surface mitigations (see `style_guide.md` §4.2).
   - Follow `style_guide.md`; when unspecified, **emulate existing codebase patterns** (APIs, errors, messaging, configuration).

5) **Performance:** focus on **worst-case latency and throughput**.
   - Apply real-time programming principles (see `style_guide.md` §4.6).

---

## When unsure
For **critical aspects**, never assume. Search the repo, then authoritative online sources; if unresolved, ask the human with a concise, pointed question.

**Critical aspects include (non-exhaustive):**
- Timing, synchronization, units, and coordinate frames. 
- Protocols and interfaces; versioning and QoS.
- Build/launch configurations that alter runtime behavior.

For **non-critical aspects**, you may make explicit assumptions. State them succinctly and **enforce them as contracts** (asserts or validation with exceptions). Keep messages short and diagnostic.

---

## Implementation discipline
- **Instruction priority:** `style_guide.md` **>** task-specific prompt **>** this document.
- **Prioritize safety and correctness** over speed of delivery.
- **Leave the codebase simpler** and more reliable than you found it.
- Prefer **pure, deterministic** core logic where feasible.
- Do not include meta-commentary or references to this document in code, comments, or commit messages.

---

## Environment assumptions

- Do not assume the environment is installed on the current machine, or the code is designed to run on the current machine.
- Never attempt to build or run code unless explicitly requested by the user.

---

## Checklist (before proposing/executing changes)
- [ ] Complies with **`style_guide.md`**.
- [ ] Uses **`requirements.md`** to validate acceptance criteria, if necessary.
- [ ] **Correctness:** contracts enforced; behavior deterministic and explicit.  
- [ ] **Simplicity:** solution is the simplest that satisfies the requirements.  
- [ ] **Maintainability:** change is understandable and auditable; comments explain why/how succinctly; no meta notes.
- [ ] **Compatibility:** integrates with existing modules/patterns; fixes underlying design flaws rather than plastering them.  
- [ ] **Performance:** considers **worst-case** latency/throughput and favors predictable behavior.
