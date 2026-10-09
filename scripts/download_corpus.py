import os
import re
import urllib.request

# Selected diverse public domain literary texts from Project Gutenberg
GUTENBERG_BOOKS = [
    {
        "id": "84",
        "title": "Frankenstein by Mary Shelley",
        "filename": "frankenstein.txt",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt"
    },
    {
        "id": "1661",
        "title": "The Adventures of Sherlock Holmes by Arthur Conan Doyle",
        "filename": "sherlock_holmes.txt",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    },
    {
        "id": "11",
        "title": "Alice's Adventures in Wonderland by Lewis Carroll",
        "filename": "alice_in_wonderland.txt",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt"
    },
    {
        "id": "35",
        "title": "The Time Machine by H.G. Wells",
        "filename": "time_machine.txt",
        "url": "https://www.gutenberg.org/cache/epub/35/pg35.txt"
    }
]

def clean_gutenberg_text(text: str) -> str:
    """Strip Gutenberg header and footer boilerplates to extract clean body text."""
    # Find start marker
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]

    # Find end marker
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]

    # Normalize line endings and excessive whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def download_corpus(output_dir: str = "data/raw") -> list[str]:
    """Download public domain books and save them into data/raw."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    print("=" * 60)
    print(" Downloading Public Domain Text Corpus (Project Gutenberg)")
    print("=" * 60)

    headers = {"User-Agent": "CustomLLMDataPipeline/1.0 (Educational Language Model Project)"}

    for book in GUTENBERG_BOOKS:
        dest_path = os.path.join(output_dir, book["filename"])
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f" [Exists] {book['title']} -> {dest_path}")
            saved_files.append(dest_path)
            continue

        print(f" [Downloading] {book['title']}...")
        try:
            req = urllib.request.Request(book["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_content = resp.read().decode("utf-8", errors="replace")

            cleaned = clean_gutenberg_text(raw_content)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"   Saved {len(cleaned):,} chars to {dest_path}")
            saved_files.append(dest_path)
        except Exception as e:
            print(f"   Warning: Could not download {book['title']}: {e}")

    # Fallback if network blocked or no files downloaded
    if not saved_files:
        print(" Creating offline sample corpus as fallback...")
        fallback_path = os.path.join(output_dir, "sample_corpus.txt")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(
                "Artificial intelligence and language models have revolutionized the way machines understand "
                "and generate human text. Language models are probabilistic models trained on large text corpora "
                "to predict the probability distribution over subsequent tokens. Using self-attention mechanisms, "
                "transformers allow deep contextual representations to be computed in parallel across sequence tokens.\n" * 500
            )
        saved_files.append(fallback_path)

    print("=" * 60)
    return saved_files

if __name__ == "__main__":
    download_corpus()
