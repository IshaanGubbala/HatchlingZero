"""Train 24K BPE tokenizer on mixed-domain corpus."""

from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


def train_bpe_tokenizer(
    corpus_path: str = "data/tokenizer_corpus/all.txt",
    output_path: str = "data/tokenizer/hz_24k.json",
    vocab_size: int = 24000,
    min_frequency: int = 2,
) -> None:
    """Train BPE tokenizer on corpus."""

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Training BPE Tokenizer (24K vocabulary)")
    print("="*70)

    # Check corpus
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        return

    corpus_size = corpus_file.stat().st_size / (1024 * 1024)
    print(f"\n✓ Corpus: {corpus_size:.1f} MB")

    # Create tokenizer
    print("\nCreating BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization (split on whitespace + punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Reserved tokens
    reserved_tokens = [
        "<|bos|>", "<|eos|>", "<|pad|>",
        "<|system|>", "<|user|>", "<|assistant|>",
        "<|tool_list|>", "<|tool_call|>", "<|tool_result|>", "<|tool_error|>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
        "<|code_start|>", "<|code_end|>",
    ]

    # Training
    print(f"Training on {corpus_path}...")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=reserved_tokens,
    )

    tokenizer.train([corpus_path], trainer=trainer)

    # Post-processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Save
    print(f"\nSaving to {output_path}...")
    tokenizer.save(output_path)

    # Statistics
    vocab = tokenizer.get_vocab()
    print(f"\n✓ Tokenizer created")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Reserved tokens: {len(reserved_tokens)}")

    # Test encoding
    print(f"\nTesting encoding...")
    test_text = "def hello(): return 'world'"
    encoded = tokenizer.encode(test_text)
    print(f"  Input: {test_text}")
    print(f"  Tokens: {encoded.tokens[:20]}")
    print(f"  IDs: {encoded.ids[:20]}")

    print("\n" + "="*70)
    print("Tokenizer training complete")
    print("="*70)


if __name__ == "__main__":
    train_bpe_tokenizer()
