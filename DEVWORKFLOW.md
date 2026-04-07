# ModelSwitch Development Workflow

This document defines the standard workflow for developing new features in ModelSwitch. Following this workflow ensures code quality, proper testing, and documentation consistency.

## Quick Reference

```
Feature Request → Analysis → Git Commit → Development → Update Docs → Run Tests → CI Passes → Review
```

---

## CI/CD

GitHub Actions CI runs automatically on push/PR to `main` with Python 3.10–3.13 matrix.

**Workflow file:** `.github/workflows/ci.yml`

```bash
# Local test before push (same as CI)
pip install pytest pytest-asyncio pytest-timeout litellm==1.82.6
python -m pytest tests/ -v --timeout=60
```

**Note:** `litellm` is pinned to 1.82.6 — do not upgrade (supply chain attack on 1.82.7/1.82.8).

---

## Workflow Steps

### 1. Feature Request & Analysis

When a new feature is requested:

1. **Understand the Request**
   - What is the feature?
   - What problem does it solve?
   - Who will use it?

2. **Analyze Code Changes**
   - Which files need modification?
   - What new files are needed?
   - What dependencies are affected?
   - Are there breaking changes?

3. **Document Analysis**
   - Create a brief summary of changes
   - List affected components
   - Identify test requirements

**Example Analysis:**
```
Feature: Add support for OpenAI o1 models with reasoning tokens

Affected Files:
- app/adapters/openai_adapter.py (add reasoning_token handling)
- app/api/openai_routes.py (parse reasoning_content)
- app/models/config_models.py (add model config for o1)
- tests/test_openai_adapter.py (new test cases)

New Files:
- None

Dependencies:
- No new dependencies

Breaking Changes:
- None
```

---

### 2. Git Preparation

**CRITICAL: Always commit current changes before starting new development.**

```bash
# Step 1: Check current status
git status

# Step 2: Stage any uncommitted changes
git add .

# Step 3: Commit with descriptive message
git commit -m "chore: checkpoint before [feature-name] development"

# Step 4: Create feature branch
git checkout -b feature/[feature-name]

# Example:
git checkout -b feature/o1-reasoning-tokens
```

**Branch Naming Convention:**
- `feature/[feature-name]` - New features
- `fix/[bug-name]` - Bug fixes
- `refactor/[component]` - Code refactoring
- `docs/[doc-name]` - Documentation updates

---

### 3. Development

#### 3.1 Write Code

- Follow existing code patterns
- Maintain consistent style
- Add inline comments for complex logic
- Handle errors gracefully

#### 3.2 Code Quality Checklist

- [ ] Code follows project style guide
- [ ] No hardcoded values (use config)
- [ ] Proper error handling
- [ ] Logging added for debugging
- [ ] No security vulnerabilities
- [ ] Type hints where applicable
- [ ] Auth consideration: Does this change affect API auth requirements?

#### 3.3 Commit During Development

Make small, focused commits during development:

```bash
# After completing a logical unit
git add [specific-files]
git commit -m "feat: add reasoning token parsing in openai_adapter"

# Commit message format:
# feat: [description] - New feature
# fix: [description] - Bug fix
# refactor: [description] - Code refactoring
# docs: [description] - Documentation
# test: [description] - Test updates
# chore: [description] - Maintenance tasks
```

---

### 4. Update Documentation

After completing development, update all relevant documentation.

#### 4.1 Update Test Cases

**Location:** `tests/test_*.py`

Add tests for:
- New functionality
- Edge cases
- Error scenarios
- Integration points

```python
# Example: Adding tests for new feature
class TestO1ReasoningTokens:
    @pytest.mark.asyncio
    async def test_reasoning_tokens_in_response(self, client):
        """Test that reasoning tokens are properly parsed."""
        # Test implementation
        
    @pytest.mark.asyncio
    async def test_reasoning_tokens_streaming(self, client):
        """Test reasoning tokens in streaming mode."""
        # Test implementation
```

#### 4.2 Update TESTGUIDE.md

Add documentation for:
- New test cases
- New test categories
- Updated coverage information
- New environment variables

**Template for TESTGUIDE.md updates:**
```markdown
### [New Feature Name]

**Test File:** `tests/test_[name].py`

| Test Name | Description |
|-----------|-------------|
| `test_[scenario]` | Description of what is tested |

**Run:**
```bash
python -m pytest tests/test_[name].py -v
```
```

#### 4.3 Update todo_test.md

Update the TODO file with:
- Completed items (move to "Done" section)
- New test requirements
- Coverage status updates

**Example update:**
```markdown
### Done

| Component | Coverage | Status |
|-----------|----------|--------|
| `o1_adapter.py` | 100% | Done (New) |

### Needs Coverage

| Component | Coverage | Missing |
|-----------|----------|---------|
| `new_module.py` | 0% | All tests needed |
```

#### 4.4 Update CLAUDE.md (if needed)

If the feature changes:
- Architecture
- Request flow
- Configuration options
- New patterns

---

### 5. Run Tests

**IMPORTANT: Always ask the user before running tests.**

#### 5.1 Ask User

```
Development complete. Would you like me to run tests?

Available test options:
1. Smoke tests only (quick validation)
2. Unit tests for changed components
3. Full test suite
4. E2E tests (requires running server)
5. Custom selection

Recommended based on your changes: [specific tests]
```

#### 5.2 Test Selection Strategy

Based on changed files, select appropriate tests:

| Changed Files | Tests to Run |
|---------------|--------------|
| `app/adapters/*.py` | Unit tests for adapter, smoke test |
| `app/api/*.py` | API integration tests, smoke test, e2e tests |
| `app/core/*.py` | Unit tests for core, smoke test |
| `app/utils/*.py` | Unit tests for utils, smoke test |
| `app/models/*.py` | Config model tests, smoke test |
| `app/services/*.py` | Service tests, smoke test |
| `config.yaml` | Smoke test, e2e tests |

#### 5.3 Always Run These Tests

Regardless of changes, **always run:**

1. **Smoke Test** - Quick validation
   ```bash
   python scripts/smoketest.py --model [model-name]
   ```

2. **E2E Tests** (if server is running)
   ```bash
   MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
   ```

#### 5.4 Run Tests Command

```bash
# 1. ALWAYS run smoke test first
python scripts/smoketest.py --model glm-5

# 2. Run unit tests for changed components
python -m pytest tests/test_[changed-component].py -v

# 3. Run API integration tests if API changed
python -m pytest tests/test_api_routes.py -v

# 4. Run E2E tests (if server running)
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v

# 5. Run full test suite with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

---

### 6. Review & Finalize

#### 6.1 Pre-Commit Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] No uncommitted changes
- [ ] Code follows style guide
- [ ] No security issues
- [ ] Coverage maintained or improved
- [ ] Auth implications documented (if changed API routes)

#### 6.2 Final Commit

```bash
# Stage all changes
git add .

# Commit with comprehensive message
git commit -m "feat: add [feature-name] with tests and documentation

- Add [component] for [functionality]
- Update [component] to support [feature]
- Add tests in test_[name].py
- Update TESTGUIDE.md and todo_test.md

Closes #[issue-number]"
```

#### 6.3 Push & Create PR

```bash
# Push to remote
git push origin feature/[feature-name]

# CI will automatically run on Python 3.10–3.13
# Check status at: https://github.com/ddmonster/modelswitch/actions

# Create pull request with:
# - Description of changes
# - Test results
# - Documentation updates
# - Breaking changes (if any)
# - CI must pass before merge
```

---

## Test Selection by Component

### Adapters (`app/adapters/`)

```bash
# Unit tests
python -m pytest tests/test_anthropic_adapter.py tests/test_openai_adapter.py -v

# Smoke test (always)
python scripts/smoketest.py --model glm-5

# E2E tests (always)
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
```

### API Routes (`app/api/`)

```bash
# Integration tests
python -m pytest tests/test_api_routes.py -v

# Smoke test (always)
python scripts/smoketest.py --model glm-5

# E2E tests (always)
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
```

### Core Components (`app/core/`)

```bash
# Unit tests
python -m pytest tests/test_chain_router.py tests/test_circuit_breaker.py tests/test_request_queue.py -v

# Smoke test (always)
python scripts/smoketest.py --model glm-5
```

### Utilities (`app/utils/`)

```bash
# Unit tests
python -m pytest tests/test_message_converter.py tests/test_usage_tracker.py -v

# Smoke test (always)
python scripts/smoketest.py --model glm-5
```

### Configuration (`app/models/`, `config.yaml`)

```bash
# Unit tests
python -m pytest tests/test_config_models.py tests/test_api_key_service.py -v

# Smoke test (always)
python scripts/smoketest.py --model glm-5

# E2E tests (always)
MODELSWITCH_E2E=1 pytest tests/test_e2e.py -v
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Feature Request                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. ANALYSIS                                                     │
│  - Understand requirements                                       │
│  - Identify affected files                                       │
│  - Document changes needed                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. GIT PREPARATION                                              │
│  - git status                                                    │
│  - git add .                                                     │
│  - git commit -m "chore: checkpoint before [feature]"           │
│  - git checkout -b feature/[name]                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. DEVELOPMENT                                                  │
│  - Write code                                                    │
│  - Follow style guide                                            │
│  - Make small commits                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. UPDATE DOCUMENTATION                                         │
│  - Add test cases (tests/test_*.py)                             │
│  - Update TESTGUIDE.md                                           │
│  - Update todo_test.md                                           │
│  - Update CLAUDE.md (if needed)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. RUN TESTS (Ask user first!)                                 │
│  - ALWAYS: smoke test                                           │
│  - ALWAYS: e2e tests (if server running)                        │
│  - SELECTIVE: unit tests for changed components                 │
│  - OPTIONAL: full test suite with coverage                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. REVIEW & FINALIZE                                            │
│  - Check pre-commit checklist                                   │
│  - Final commit                                                  │
│  - Push and create PR                                            │
│  - CI runs automatically (Python 3.10–3.13)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Emergency Procedures

### Tests Failing

1. **Don't panic** - analyze the failure
2. Check if it's related to your changes
3. If unrelated, document and proceed
4. If related, fix and re-run

### Need to Abort Development

```bash
# Save current work
git add .
git commit -m "wip: [feature-name] in progress"

# Return to main branch
git checkout main

# Resume later
git checkout feature/[feature-name]
```

### Accidental Changes to Wrong Branch

```bash
# Stash changes
git stash

# Switch to correct branch
git checkout feature/[correct-name]

# Apply stashed changes
git stash pop
```

---

## Summary Checklist

Before considering development complete:

- [ ] Analysis documented
- [ ] Git checkpoint created
- [ ] Feature branch created
- [ ] Code developed
- [ ] Test cases added
- [ ] TESTGUIDE.md updated
- [ ] todo_test.md updated
- [ ] User asked about running tests
- [ ] Smoke test passed
- [ ] E2E tests passed (if applicable)
- [ ] Unit tests passed
- [ ] Coverage maintained/improved
- [ ] Final commit made
- [ ] Push to remote
- [ ] CI passes (Python 3.10–3.13)
- [ ] PR created

---

**Remember:** Always ask the user before running tests. Smoke tests and E2E tests are mandatory for all changes.