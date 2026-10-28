from collections import defaultdict


def _pre_tokens_to_bytes(pre_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """Return dict with each unique tuple of bytes and its frequency."""
    d = defaultdict(int)

    for s in pre_tokens:
        enc = s.encode(encoding="utf-8")
        tuple_bytes = tuple([bytes([val]) for val in enc])  # [b] is the trick.

        d[tuple_bytes] += 1

    return d


def _get_pair_counts(d_freq_unique_bytes: dict[tuple[bytes, ...], int]):
    """Using d_unique_pretok. Get pairs of bytes, and frequency."""

    d_byte_pairs_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)

    for k, v in d_freq_unique_bytes.items():
        for idx in range(len(k) - 1):
            curr_bt = k[idx]
            next_bt = k[idx + 1]

            d_byte_pairs_freq[(curr_bt, next_bt)] += v

    return d_byte_pairs_freq


def _merge_2(bt: tuple[bytes, ...], most_freq: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Iterate through a tuple of bytes until the "most_freq" are found, and merge them."""
    res = ()

    idx = 0
    len_bt = len(bt[:-1])

    # iterate through bytes until the "most_freq" are found, and merge them.
    while idx < len_bt:
        curr_bt = bt[idx]
        next_bt = bt[idx + 1]

        if (curr_bt, next_bt) == most_freq:
            new_b = curr_bt + next_bt
            res += (new_b,)
            idx += 2

        else:
            res += (curr_bt,)
            idx += 1

    # print(f"{idx=}, {len_t=}")
    while idx <= len_bt:
        curr_bt = bt[idx]
        res += (curr_bt,)
        idx += 1

    return res


def pug_bpe(
    pre_tokens: list[str], vocab: dict[int, bytes], max_vocab_size: int = 6
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Implementation of BPE Tokenizer training."""

    # Just once iterate through all pre_tokens. Return unique bytes with their frequencies.
    d_freq_unique_pretok = _pre_tokens_to_bytes(pre_tokens)

    merges: list[tuple[bytes, bytes]] = []
    len_vocab = len(vocab)

    while len_vocab < max_vocab_size:

        d_byte_pairs_freq = _get_pair_counts(d_freq_unique_pretok)

        most_freq = tuple()
        max_ = -1
        for k, v in d_byte_pairs_freq.items():
            if v >= max_:
                if v == max_:
                    most_freq = max(most_freq, k)  # greatest lexicographically

                else:  # v > max_
                    most_freq = k

                max_ = v

        if not most_freq or len(most_freq) < 2:  # nothing relevant left to merge
            break

        merges.append(most_freq)

        curr_b, next_b = most_freq
        res_b = curr_b + next_b
        vocab[len_vocab] = res_b

        updated_d_freq_unique_pretok: dict[tuple[bytes, ...], int] = {}
        for k, v in d_freq_unique_pretok.items():
            new_k = _merge_2(k, most_freq)

            updated_d_freq_unique_pretok[new_k] = v

        d_freq_unique_pretok = updated_d_freq_unique_pretok.copy()

        len_vocab += 1

    return vocab, merges
