class NounFST:
    def __init__(self, lexicon_file=None, custom_lexicon=None):
        self.vowels = {'a', 'e', 'i', 'o', 'u'}
        self.sibilants = ('s', 'z', 'x', 'ch', 'sh')
        
        # Load lexicon from file or set
        if custom_lexicon:
            self.lexicon = set(custom_lexicon)
        elif lexicon_file:
            with open(lexicon_file, 'r') as f:
                self.lexicon = set(line.strip().lower() for line in f if line.strip())
        else:
            # Fallback sample lexicon from Brown Corpus
            self.lexicon = {"fox", "watch", "baby", "bag", "boy", "cat", "dog", "city", "bus", "quiz"}

    def parse(self, surface_word: str) -> str:
        word = surface_word.lower().strip()

        # State 1: Check Singular (Direct lexicon lookup)
        if word in self.lexicon:
            return f"{word} = {word}+N+SG"

        # State 2: Check Plural Form & Rule Validations
        
        # Rule 1: E-Insertion (-es ending)
        if word.endswith('es'):
            # Sibilant root check (e.g., foxes -> fox, watches -> watch)
            stem_candidate = word[:-2]
            if any(stem_candidate.endswith(sib) for sib in self.sibilants):
                if stem_candidate in self.lexicon:
                    return f"{word} = {stem_candidate}+N+PL"
            
            # Stems natively ending in 'e' (e.g., gates -> gate)
            e_stem_candidate = word[:-1]
            if e_stem_candidate in self.lexicon:
                return f"{word} = {e_stem_candidate}+N+PL"

        # Rule 2: Y-Replacement (-ies ending for consonant + y)
        if word.endswith('ies'):
            stem_candidate = word[:-3] + 'y'
            if len(stem_candidate) >= 2 and stem_candidate[-2] not in self.vowels:
                if stem_candidate in self.lexicon:
                    return f"{word} = {stem_candidate}+N+PL"

        # Rule 3: S-Addition (-s or -ys ending)
        if word.endswith('s'):
            # Vowel + y + s (e.g., boys -> boy)
            if word.endswith('ys'):
                stem_candidate = word[:-1]
                if len(stem_candidate) >= 2 and stem_candidate[-2] in self.vowels:
                    if stem_candidate in self.lexicon:
                        return f"{word} = {stem_candidate}+N+PL"
            
            # Standard -s addition
            elif not word.endswith('es'):
                stem_candidate = word[:-1]
                if stem_candidate in self.lexicon:
                    # REJECT if root ends in sibilant (must use -es, e.g., foxs is invalid)
                    if not any(stem_candidate.endswith(sib) for sib in self.sibilants):
                        return f"{word} = {stem_candidate}+N+PL"

        # State 3: Reject State
        return f"{word} = Invalid Word"


# --- Execution Example ---
if __name__ == "__main__":
    fst = NounFST()

    test_words = [
        "fox", "foxes", "foxs",       # E insertion rule tests
        "watch", "watches", "watchs", # E insertion rule tests
        "baby", "babies", "babys",   # Y replacement rule tests
        "boy", "boys", "boies",       # Vowel + Y tests
        "bag", "bags", "bages"        # S addition tests
    ]

    for word in test_words:
        print(fst.parse(word))