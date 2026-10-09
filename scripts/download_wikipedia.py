import os
import re
import json
import time
import urllib.request
import urllib.parse

# Curated list of high-quality modern Wikipedia articles across diverse domains
WIKIPEDIA_ARTICLES = [
    # Computer Science & AI
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Computer",
    "Internet",
    "Algorithm",
    "Python (programming language)",
    "Software engineering",
    "Robotics",

    # Physics & Astronomy
    "Solar System",
    "Mars",
    "Black hole",
    "Quantum mechanics",
    "General relativity",
    "Speed of light",
    "Thermodynamics",
    "Gravitational wave",

    # Biology, Genetics & Medicine
    "DNA",
    "Gene",
    "Cell (biology)",
    "Evolution",
    "Photosynthesis",
    "Human brain",
    "Neuron",
    "Immune system",

    # Energy, Technology & Exploration
    "Electricity",
    "Renewable energy",
    "Nuclear power",
    "Space exploration",
    "Apollo program",
    "Electric vehicle",

    # History & Civilization
    "Industrial Revolution",
    "World War II",
    "Renaissance",
    "Scientific Revolution",
    "Philosophy of science"
]

def clean_wiki_text(text: str) -> str:
    """Cleans up Wikipedia extract text by removing excessive section headers and blank lines."""
    # Remove empty sections or Wikipedia reference markers
    text = re.sub(r"==+\s+See also\s+==+.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"==+\s+References\s+==+.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"==+\s+Further reading\s+==+.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"==+\s+External links\s+==+.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Clean section headers: === Header === -> Header:
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\n\1:\n", text)
    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def download_wikipedia_corpus(output_dir: str = "data/raw") -> list[str]:
    """Downloads modern Wikipedia articles as clean text files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    print("=" * 65)
    print(f" Downloading Modern Wikipedia Corpus ({len(WIKIPEDIA_ARTICLES)} Articles)")
    print("=" * 65)

    headers = {"User-Agent": "CustomLLMModernTraining/1.0 (Educational research project)"}

    total_chars = 0
    for i, title in enumerate(WIKIPEDIA_ARTICLES, 1):
        safe_name = re.sub(r"[^\w\-_]", "_", title.lower()) + ".txt"
        dest_path = os.path.join(output_dir, safe_name)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 2000:
            sz = os.path.getsize(dest_path)
            total_chars += sz
            print(f" [{i:02d}/{len(WIKIPEDIA_ARTICLES)}] [Cached] {title:<32} ({sz:,} chars)")
            saved_files.append(dest_path)
            continue

        encoded_title = urllib.parse.quote(title)
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&titles={encoded_title}&format=json"
        )

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                extract = page.get("extract", "")
                if extract:
                    cleaned = clean_wiki_text(extract)
                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    total_chars += len(cleaned)
                    print(f" [{i:02d}/{len(WIKIPEDIA_ARTICLES)}] [Downloaded] {title:<28} ({len(cleaned):,} chars)")
                    saved_files.append(dest_path)
                    break
            time.sleep(1.2)  # Respect rate limits
        except Exception as e:
            print(f" [{i:02d}/{len(WIKIPEDIA_ARTICLES)}] [Warning] Failed {title}: {e}")

    print("=" * 65)
    print(f"Downloaded {len(saved_files)} articles | Total text: {total_chars:,} characters (~{total_chars / 1024 / 1024:.2f} MB)")
    print("=" * 65)
    return saved_files

if __name__ == "__main__":
    download_wikipedia_corpus()
