# ANN Iris Streamlit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hosted Streamlit dashboard for the ANN Iris classifier project.

**Architecture:** Put reusable model and data functions in `iris_model.py`, render the dashboard in `app.py`, and document GitHub plus Streamlit Cloud deployment in `README.md`.

**Tech Stack:** Python, Streamlit, scikit-learn, pandas, matplotlib, pytest.

---

### Task 1: Model Helpers

**Files:**
- Create: `iris_model.py`
- Create: `tests/test_iris_model.py`

- [ ] Write tests for dataset loading, model training, and prediction probabilities.
- [ ] Run pytest and confirm the tests fail before implementation.
- [ ] Implement the smallest model helpers that pass.
- [ ] Run pytest again and confirm the tests pass.

### Task 2: Streamlit Dashboard

**Files:**
- Create: `app.py`
- Create: `requirements.txt`

- [ ] Build the learning dashboard selected by the user.
- [ ] Add dependency pins compatible with Streamlit Cloud.
- [ ] Compile the Python files to catch syntax errors.

### Task 3: GitHub Documentation

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Copy: `ANN(1).ipynb`

- [ ] Explain project purpose, features, local run steps, and Streamlit Cloud hosting steps.
- [ ] Ignore virtual environments, caches, notebook checkpoints, and local Streamlit config.
- [ ] Stage only intended files, commit, and push to `origin/main`.
