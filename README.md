# Exploit Mitigation Techniques: Performance vs. Security Trade-offs

This project explores the performance impact and security benefits of various exploit mitigation techniques like **Address Space Layout Randomization (ASLR)**, **Data Execution Prevention (DEP)**, and **Control Flow Integrity (CFI)**. We evaluate these techniques using test programs and benchmark the performance with and without these protections.

---

## 🚀 Objectives

- Understand how modern operating systems prevent exploitation using ASLR, DEP, and CFI.
- Measure performance overhead of each mitigation.
- Compare trade-offs between security and efficiency.

---

## 🛡️ Techniques Covered

### 1. ASLR (Address Space Layout Randomization)
Randomizes memory layout to prevent predictable code injection/execution.

### 2. DEP (Data Execution Prevention)
Prevents code execution from non-executable memory regions (e.g., stack, heap).

### 3. CFI (Control Flow Integrity)
Restricts indirect control flow to only valid call targets.

---

