# Contributing to Custom LLM

First off, thank you for considering contributing to **Custom LLM**! Open-source thrives on community contributions, whether it's reporting a bug, improving documentation, submitting optimizations, or adding new features.

---

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## How Can I Contribute?

### 1. Reporting Bugs
If you encounter a bug:
1. Search the [Issues](https://github.com/milan/custom-llm/issues) tab to see if it has already been reported.
2. If not, open a new issue with:
   - A clear and descriptive title.
   - Exact steps to reproduce the issue.
   - Your environment specifications: OS, Python version, PyTorch version, GPU model (if applicable), and CUDA version.
   - Full terminal error traceback.

### 2. Suggesting Enhancements
Feature requests and optimizations are very welcome!
- Describe the proposed enhancement and the motivation behind it.
- Explain alternative solutions or features you considered.
- Provide code snippets, pseudocode, or architecture diagrams if helpful.

### 3. Submitting Code (Pull Requests)

#### Step 1: Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/custom-llm.git
cd custom-llm
```

#### Step 2: Set Up Development Environment
Create and activate a virtual environment:
```bash
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
# With CUDA support:
pip install -r requirements.txt

# Or CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers numpy matplotlib tqdm
```

#### Step 3: Verify Your Setup
Run the diagnostic and model verification scripts:
```bash
python scripts/verify_cuda.py
python scripts/verify_model.py
```
Both scripts should pass with status `[SUCCESS]`.

#### Step 4: Create a Feature Branch
```bash
git checkout -b feat/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

#### Step 5: Implement Your Changes & Keep Repository Clean
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Maintain existing comments and document any new functions or classes.
- **IMPORTANT**: **Never commit binary model checkpoints (`*.pt`, `*.bin`, `*.safetensors`) or dataset binary caches to git**. All `.gitignore` rules must be strictly respected.

#### Step 6: Test Your Changes
Make sure the GUI and CLI inference function correctly:
```bash
# Test CLI inference
python scripts/inference.py --prompt "Hello, world!" --max_tokens 20

# Test GUI launch
python scripts/gui_app.py
```

#### Step 7: Commit and Push
Write clear, imperative commit messages:
```bash
git add .
git commit -m "feat(attention): optimize flash attention rotary embeddings"
git push origin feat/your-feature-name
```

#### Step 8: Open a Pull Request
Go to the repository on GitHub and click **Compare & pull request**. Provide a clear description of the problem solved and the implementation details.

---

## Coding Conventions

- **Python Version**: Target Python 3.10+ (tested up to 3.12).
- **Docstrings**: Use clear docstrings describing arguments, return values, and shapes of PyTorch tensors (e.g., `(batch_size, seq_len, hidden_dim)`).
- **Error Handling**: Provide helpful, human-readable error messages for missing checkpoints, missing CUDA drivers, or corrupted data.
- **Cross-Platform Compatibility**: Use `os.path` or `pathlib.Path` for path manipulations to ensure scripts run seamlessly on Windows, Linux, and macOS.

Thank you for contributing! 🚀
