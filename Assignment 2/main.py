import json
import math

# ============================================================
# LOAD DATASET
# ============================================================

with open("text_segmentation_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
word_counts = data["word_counts"]
test_cases = data["test_cases"]

total_corpus_words = data["metadata"]["total_corpus_words"]

vocabulary = set(word_counts.keys())


# ============================================================
# LOG PROBABILITIES
# ============================================================

log_prob = {}

for word, count in word_counts.items():
    probability = count / total_corpus_words
    log_prob[word] = math.log(probability)


# Maximum word length 
MAX_WORD_LENGTH = max(len(word) for word in vocabulary)


# ============================================================
# 1. GREEDY LONGEST MATCH
# ============================================================

def greedy_segment(text):

    result = []

    i = 0
    n = len(text)

    while i < n:

        best_word = None

        # Try longest words first
        max_end = min(n, i + MAX_WORD_LENGTH)

        for j in range(i + 1, max_end + 1):

            candidate = text[i:j]

            if candidate in vocabulary:
                best_word = candidate

        if best_word is None:

            # Unknown character
            result.append(text[i])
            i += 1

        else:

            result.append(best_word)
            i += len(best_word)

    return result


# ============================================================
# 2. DYNAMIC PROGRAMMING
# ============================================================

def dp_segment(text):

    n = len(text)

    # dp[i] = best log probability for text[:i]
    dp = [-float("inf")] * (n + 1)

    # best_word[i] = word used to reach position i
    best_word = [None] * (n + 1)

    dp[0] = 0

    for i in range(n):

        if dp[i] == -float("inf"):
            continue

        max_end = min(n, i + MAX_WORD_LENGTH)

        for j in range(i + 1, max_end + 1):

            word = text[i:j]

            if word not in vocabulary:
                continue

            score = dp[i] + log_prob[word]

            if score > dp[j]:

                dp[j] = score
                best_word[j] = word

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    result = []

    i = n

    while i > 0:

        word = best_word[i]

        if word is None:

            result.append(text[i - 1])
            i -= 1

        else:

            result.append(word)
            i -= len(word)

    result.reverse()

    return result


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):

    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty string cases
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if a[i - 1] == b[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]

            else:

                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1]    # substitution
                )

    return dp[m][n]


# ============================================================
# EVALUATION
# ============================================================

greedy_correct_words = 0
dp_correct_words = 0

total_words = 0

greedy_edit_distance = 0
dp_edit_distance = 0


for case in test_cases:

    text = case["input"]

    # Ground truth string → list of words
    ground_truth = case["ground_truth"].split()

    # --------------------------------------------------------
    # Greedy
    # --------------------------------------------------------

    greedy_prediction = greedy_segment(text)

    # --------------------------------------------------------
    # DP
    # --------------------------------------------------------

    dp_prediction = dp_segment(text)

    # --------------------------------------------------------
    # Word accuracy
    # --------------------------------------------------------

    # Compare complete sequences using edit-distance alignment
    # for counting correctly predicted words.
    
    greedy_correct_words += sum(
        1
        for predicted, actual in zip(
            greedy_prediction,
            ground_truth
        )
        if predicted == actual
    )

    dp_correct_words += sum(
        1
        for predicted, actual in zip(
            dp_prediction,
            ground_truth
        )
        if predicted == actual
    )

    total_words += len(ground_truth)

    # --------------------------------------------------------
    # Edit Distance
    # --------------------------------------------------------

    greedy_edit_distance += edit_distance(
        greedy_prediction,
        ground_truth
    )

    dp_edit_distance += edit_distance(
        dp_prediction,
        ground_truth
    )


# ============================================================
# FINAL METRICS
# ============================================================

greedy_accuracy = greedy_correct_words / total_words

dp_accuracy = dp_correct_words / total_words


average_greedy_edit_distance = (
    greedy_edit_distance / len(test_cases)
)

average_dp_edit_distance = (
    dp_edit_distance / len(test_cases)
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 55)
print("TEXT SEGMENTATION RESULTS")
print("=" * 55)

print()

print(f"Number of test cases : {len(test_cases)}")
print(f"Total words         : {total_words}")

print()

print("GREEDY LONGEST MATCH")
print("-" * 30)

print(f"Correct words       : {greedy_correct_words}")
print(f"Accuracy            : {greedy_accuracy:.4f}")
print(f"Accuracy (%)        : {greedy_accuracy * 100:.2f}%")
print(f"Total Edit Distance : {greedy_edit_distance}")
print(f"Average Edit Dist.  : {average_greedy_edit_distance:.4f}")

print()

print("DYNAMIC PROGRAMMING")
print("-" * 30)

print(f"Correct words       : {dp_correct_words}")
print(f"Accuracy            : {dp_accuracy:.4f}")
print(f"Accuracy (%)        : {dp_accuracy * 100:.2f}%")
print(f"Total Edit Distance : {dp_edit_distance}")
print(f"Average Edit Dist.  : {average_dp_edit_distance:.4f}")

print("=" * 55)

# ============================================================
# SAMPLE INPUT / OUTPUT
# ============================================================

sample_input = "thegovernmentforallitsworkers"

sample_greedy = greedy_segment(sample_input)
sample_dp = dp_segment(sample_input)

print()
print("=" * 55)
print("SAMPLE INPUT / OUTPUT")
print("=" * 55)

print()
print("Sample Input:")
print(sample_input)

print()
print("Greedy Longest Match:")
print(" ".join(sample_greedy))

print()
print("Dynamic Programming:")
print(" ".join(sample_dp))

print()
print("Ground Truth:")
print("the government for all its workers")

print("=" * 55)