class VanillaBPE:
    def __init__(self, target_vocab_amount):
        self.target_vocab_amount = target_vocab_amount
        self.saved_merges = {}
        self.vocabulary = {}

    def count_adjacent_pairs(self, list_of_numbers):
        pair_counts = {}
        for pair in zip(list_of_numbers, list_of_numbers[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def replace_pair_with_new_number(self, list_of_numbers, pair_to_find, new_number):
        if pair_to_find[0] not in list_of_numbers:
            return list_of_numbers

        new_list = []
        idx = 0
        while idx < len(list_of_numbers):
            if idx < len(list_of_numbers) - 1 and list_of_numbers[idx] == pair_to_find[0] and list_of_numbers[idx + 1] == pair_to_find[1]:
                new_list.append(new_number)
                idx += 2
            else:
                new_list.append(list_of_numbers[idx])
                idx += 1
        return new_list

    def train_the_model(self, my_text):
        tokens = list(my_text.encode("utf-8"))

        while 256 + len(self.saved_merges) < self.target_vocab_amount:
            stats = self.count_adjacent_pairs(tokens)
            if not stats:
                print("No more pairs left to merge!")
                break

            best_pair = max(stats, key=stats.get)
            new_token_id = 256 + len(self.saved_merges)
            self.saved_merges[best_pair] = new_token_id
            tokens = self.replace_pair_with_new_number(tokens, best_pair, new_token_id)
            print("Merged", best_pair, "into", new_token_id)

        self.vocabulary = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.saved_merges.items():
            self.vocabulary[new_id] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]

    def encode_text(self, my_text):
        tokens = list(my_text.encode("utf-8"))

        while True:
            stats = self.count_adjacent_pairs(tokens)
            known_pairs = {p: self.saved_merges[p] for p in stats if p in self.saved_merges}
            if not known_pairs:
                break

            best_pair = min(known_pairs, key=known_pairs.get)
            tokens = self.replace_pair_with_new_number(tokens, best_pair, known_pairs[best_pair])

        return tokens

    def decode_text(self, ids):
        return b"".join(self.vocabulary.get(i, b"") for i in ids).decode("utf-8", errors="replace")
