import torch

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus using PyTorch for pair counting.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    words = [key.split() for key in corpus.keys()]
    frequencies = torch.tensor(list(corpus.values()), dtype=torch.long)

    merges = []

    for _ in range(num_merges):

        # 1. Count adjacent pairs
        pair_counts = {}

        for tokens, freq in zip(words, frequencies):

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + freq.item()
                )

        # No possible merge
        if not pair_counts:
            break

        # 2. Find most frequent pair
        best_pair = max(
            pair_counts,
            key=pair_counts.get
        )

        # 3. Record merge
        merges.append(best_pair)

        # 4. Merge pair everywhere
        new_words = []

        for tokens in words:

            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    # Merge tokens
                    new_tokens.append(
                        tokens[i] + tokens[i + 1]
                    )

                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_words.append(new_tokens)

        words = new_words

    return merges