import random

class AdvancedTokenizer:
    def __init__(self, target_vocab_size, dropout_prob=0.1):
        self.vocab_size = target_vocab_size
        self.dropout_prob = dropout_prob
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, ids):
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _apply_merge(self, ids, target_pair, new_id):
        if target_pair[0] not in ids:
            return ids

        result = []
        idx = 0
        while idx < len(ids):
            if idx < len(ids) - 1 and ids[idx] == target_pair[0] and ids[idx + 1] == target_pair[1]:
                result.append(new_id)
                idx += 2
            else:
                result.append(ids[idx])
                idx += 1
        return result

    def train(self, text):
        ids = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = self._apply_merge(ids, best_pair, new_id)
            print(f"Step {i + 1}: Merged {self.vocab[best_pair[0]]} + {self.vocab[best_pair[1]]} -> {self.vocab[new_id]}")

    def encode(self, text, apply_dropout=False):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            eligible_rules = {p: self.merges[p] for p in stats if p in self.merges}
            if not eligible_rules:
                break

            merged_any = False
            for pair, target_id in sorted(eligible_rules.items(), key=lambda item: item[1]):
                if apply_dropout and random.random() < self.dropout_prob:
                    continue
                ids = self._apply_merge(ids, pair, target_id)
                merged_any = True
                break

            if not merged_any:
                break

        return ids

    def decode(self, ids):
        return b"".join(self.vocab.get(i, b"") for i in ids).decode("utf-8", errors="replace")