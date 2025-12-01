# Style Guide
This guide defines coding standards for human developers and LLMs. LLMs must also follow `llm.md`.

## 1. Consistency & Terminology
1. Use **one canonical term per concept** across the repo. Do not mix synonyms (e.g., use `index`, never `idx`).
2. Prefer full words over truncations unless an abbreviation meets section 2.6 criteria.
3. Maintain ASCII-only source (identifiers, comments, docs). Do not use Unicode symbols such as Greeks or emojis.

## 2. Naming
### 2.1 Identifiers (non-types)
1. Use `lower_snake_case` for variables, functions, and methods.
2. Names must be descriptive. Avoid single-letter names except conventional math symbols in tight scope (section 2.7).
3. Loop counters must be meaningful: use `sample_index`, `wheel_index` (never `i`, `j`, etc.).
4. Include units for quantities with units: `length_m`, `angle_rad`, `rate_hz`.
5. Time quantities: `_period_s` for cyclic durations (e.g., `control_period_s`), `_duration_s` for non-cyclic time spans (e.g., `prediction_duration_s`), `_timestep_s` for integration step sizes.

### 2.2 Types
1. Use `PascalCase` for classes/structs/type aliases (C++ and Python).

### 2.3 Constants
1. Use `UPPER_SNAKE_CASE` for constants.
2. Numeric literals must be named constants, except `-1`, `0`, `1`, `2`, where idiomatic and unambiguous.
3. Encode units in constant names when applicable.

### 2.4 C++ Member Data
1. Private members use a trailing underscore (e.g., `speed_mps_`).
2. Do not use `m_` or `s_` prefixes.
3. Favor accessors over public data members.

### 2.5 Files & Modules
1. C++/Python files: `lower_snake_case`. Tests: `*_test.cpp`, `test_*.py`.
2. Python: public API has no leading underscore; private helpers start with `_`. Expose the public surface via `__all__`.

### 2.6 Abbreviations
#### 2.6.1 Acceptance Criteria
Use an abbreviation only if all are true:
1. **Domain-standard or industry-standard** (robotics/controls or general computing).
2. **Unambiguous** within the codebase and documentation.
3. **Consistently capitalized** as an acronym (e.g., `IMU`, `MPC`) or a widely accepted short word (`config`, `init`, `param`).
4. First use in docs/comments expands the term when not obvious.

#### 2.6.2 Allowed (non-exhaustive)
- Robotics/controls: `MPC`, `PID`, `IMU`, `ROS`, `QP`, `NLP`, `LQR`, `EKF`, `UKF`, `SLAM`.
- General tech: `API`, `CPU`, `GPU`, `GUI`, `UUID`/`ID`, `TCP`, `UDP`, `FIFO`, `PWM`, `GPS`.
- Common setup words: `init`, `config`, `param`.
- Units as suffixes: `_ms`, `_s`, `_m`, `_cm`, `_deg`, `_rad`, `_Hz`.
- Math symbols when conventional and local.

#### 2.6.3 Prohibited (non-exhaustive)
- `OCP` -> `optimal_control_problem`
- `traj` -> `trajectory`
- `vel` -> `velocity`
- `pos` -> `position`
- `ctrl` -> `control`/`controller`
- `calc` -> `calculate`
- `temp` -> `temporary` or `temperature`
- `msg` -> `message`
- Loop indices `i`, `j`, `k`; short forms like `*_idx` -> use `*_index`

#### 2.6.4 External Dependency Interface Exception
When an external dependency interface uses a prohibited abbreviation, use the abbreviation **only at the interface boundary**. Spell out the full term everywhere else:
- In our code (variable names, function names, comments)
- In documentation
- Example: ACADOS uses `ocp` and `rti` in its API. Use `AcadosOcp()` when calling the library, but name your variables `optimal_control_problem` and document "Real-Time Iteration (RTI) mode" in comments.

### 2.7 Math Symbols
Single-letter symbols may be used when they are conventional in the algorithm and **tightly scoped** (e.g., within a short function or loop). Otherwise, prefer descriptive names. Use consistent mappings:
| Symbol | Meaning | Code name pattern |
|-------:|---------|-------------------|
| `x` | state vector | `state` or member `state_` |
| `u` | control input | `control_input` or `control_input_` |
| `A` | state transition matrix | `state_matrix` or `state_matrix_` |
| `B` | input matrix | `control_matrix` or `control_matrix_` |
| `Q` | state cost matrix | `state_cost` or `state_cost_` |
| `R` | control cost matrix | `control_cost` or `control_cost_` |
| `P` | covariance matrix | `covariance` or `covariance_` |
| `F` | state Jacobian | `state_jacobian` or `state_jacobian_` |
| `H` | measurement matrix | `measurement_matrix` or `measurement_matrix_` |
| `y` | measurement | `measurement` or `measurement_` |
| `dt` | time step | `timestep` or `timestep_` |

### 2.8 Integer Types
1. **C++:** Use explicit-width types: `int32_t`, `uint32_t`, `int64_t`, `uint64_t` (from `<cstdint>`).
2. Use `size_t` for array indices, container sizes, and iterator offsets.
3. Never use `int`, `long`, `unsigned int`, `unsigned long` due to platform-dependent sizes.
4. **Python:** Use built-in `int` (arbitrary precision); annotate with `int` in type hints.

### 2.9 C++ Auto Keyword
1. **Guideline:** Use auto for mechanical type deduction; use explicit types when type conveys meaning. When in doubt, prefer explicit types.
2. **Use `auto` for:**
   - Container iterators: `auto it = map.find(key);`
   - Lambda expressions
   - Abbreviated function templates: `void f(const auto& param);`
   - Smart pointer factories where type repeats: `auto ptr = std::make_unique<Type>();`
   - Complex template metaprogramming types
3. **Avoid `auto` for:**
   - Numeric types and casts (int32_t, size_t, double, etc.)
   - Loop counters where type semantics matter
   - Function parameters (non-template)
   - When type conveys semantic information
   - Range-based for loops (shows element type and copy vs reference)
   - Pointers at API boundaries (e.g., C-style pointers for C APIs)

## 3. File & Module Organization
### 3.1 C++ Headers (`.hpp`)
1. Start with `#pragma once`.
2. Declare interfaces; keep implementations out of headers (except templates).
3. Minimize includes; prefer forward declarations.
4. Document public APIs with Doxygen.

### 3.2 Include Order
1. Corresponding header.
2. C system headers (e.g., `<cstdint>`, `<cmath>`).
3. C++ standard library (e.g., `<vector>`, `<memory>`, `<algorithm>`).
4. Third-party libraries (e.g., ROS 2, Eigen, Acados).
5. Project headers.

### 3.3 C++ Implementations (`.cpp`)
1. Keep implementation details here; headers remain minimal.

### 3.4 Python Packages
1. Public vs private as in section 2.5; internal modules under `_internal/`.
2. Module docstrings state purpose, inputs, and outputs.

## 4. Architecture
### 4.1 Separation of Concerns
1. Keep modules/classes single-purpose; avoid names with `And` or `Manager`—reconsider the design.
2. Respect interface vs implementation boundaries (C++ headers minimal; Python public API clear).
3. Aim for high cohesion within modules/classes and low coupling between them; related functionality should be grouped together, and dependencies between modules should be minimal and explicit.

### 4.2 Dependency Management
1. Include/import only what you use; avoid transitive dependencies.
2. Use dependency injection: pass collaborators via constructors/parameters; depend on interfaces/ABCs (or templates) rather than concrete types.
3. Avoid global/singleton state; pass context explicitly.
4. Keep interfaces narrow and stable.
5. Reuse before reinventing; fit into existing modules and contracts.
6. If an existing interface or design is flawed, prioritize fixing the underlying issue over surface mitigations.

### 4.3 Error Handling & Logging
1. Enforce runtime contracts at module and interface boundaries: validate preconditions, postconditions, and invariants.
2. On unrecoverable violation, fail fast with clear, diagnostic messages.
3. Apply defensive programming principles: always validate state and inputs.
4. Follow repository conventions consistently. If absent:
   - Python: raise specific exceptions; do not return sentinel values for error signaling.
   - C++: prefer a result+error mechanism (`std::expected<T,E>` or project equivalent). If unavailable, document error contracts clearly.
5. Log at appropriate levels; no logging in tight inner loops unless essential. Library code should not print user-visible output.

### 4.4 Configuration & Parameters
1. **Single source of truth**: Each parameter has exactly one authoritative definition.
2. Configuration files (YAML, JSON, etc.): No defaults in code; code reads and validates. Fail fast if required parameters are missing or invalid.
3. In-code constants: Allowed when by design not in config files; these are the single source of truth, not defaults.

### 4.5 Determinism & State
1. Prefer deterministic behavior; avoid hidden state and nondeterminism.
2. Make state management explicit; document state transitions and invariants.

### 4.6 Performance & Real-Time
1. Apply real-time programming principles for time-critical code.
2. Treat deadlines as hard constraints.
3. Optimize for worst-case latency and throughput, not average speed.
4. Focus on predictability and bounded behavior.

## 5. Function & Class Design
### 5.1 Functions/Methods
1. Single responsibility; one abstraction level per function.
2. Nesting depth ≤ 3; prefer early returns.
3. Parameter clarity: avoid ambiguous booleans controlling multiple behaviors; use enums or separate functions.
4. **Python:** Use `@property` decorator for zero-argument getters that are side-effect-free. Properties must be cheap to compute and idempotent.

### 5.2 Classes
1. Single responsibility; compose rather than inherit unless true subtyping.
2. Provide abstract interfaces for swappable components; inject them where used.

## 6. Clarity
1. Clarity over brevity in names and control flow.
2. No magic numbers. Define named constants for all domain literals and unit conversions (allowed inline literals: `-1`, `0`, `1`, `true`, `false`).
3. Encode units in names when relevant.
4. Keep comments high-value: why/how > what. Avoid restating the code.
5. Do not duplicate requirements, configuration values, or specifications in comments. Reference the authoritative source (e.g., "See requirements.md section 2.2.1") instead of hardcoding values. Configuration-dependent behavior should read from config, not document hardcoded assumptions.

## 7. Checklist
- [ ] **Consistency:** single canonical term per concept (e.g., `index`, not `idx`).
- [ ] **Naming:** `snake_case` for non-types; `PascalCase` for types; constants `UPPER_SNAKE_CASE`; units in names where helpful. Meaningful counters, no `i, j` for loops.
- [ ] **Abbreviations:** follows section 2.6; allowed lists only; prohibited forms spelled out; external dependency abbreviations used only at interface boundary.
- [ ] **Math symbols:** follows section 2.7; single-letter symbols only when conventional and local.
- [ ] **C++ Auto keyword:** follows section 2.9; auto for machinery (iterators, lambdas, templates), explicit types for meaning (numeric types, semantics, API boundaries).
- [ ] **C++ headers:** correct order; `#pragma once`; minimal; forward declarations; public API documented.
- [ ] **Python includes:** correct order; no unused includes/imports.
- [ ] **Implementation:** implementation and local declarations in `.cpp`, not `.hpp`; Python public vs private respected; `_internal_name` for internals.
- [ ] **Architecture:** dependency injection used; reuse before reinvent; no globals/singletons; narrow interfaces; single source of truth for parameters.
- [ ] **Contracts & errors:** runtime contracts enforced at boundaries; fail fast on violations; defensive programming; informative error messages.
- [ ] **Determinism:** deterministic behavior preferred; explicit state management; no hidden state.
- [ ] **Performance:** real-time principles applied where needed; worst-case behavior considered; predictable and bounded.
- [ ] **Clarity:** no magic numbers; named constants; comments explaining "why" and "how.
- [ ] **Text encoding:** ASCII-only. No Unicode symbols. No Greeks or emojis.
