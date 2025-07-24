from collections import Counter
import string

text = "I love learning. I love Python. Python is fun!"

# 1. lowercase + remove punctuation
text = text.lower().translate(str.maketrans("", "", string.punctuation))
words = text.split()

# 2. remove stopwords
stopwords = ['i', 'is', 'the', 'and', 'a']
filtered = [w for w in words if w not in stopwords]

# 3. count top words
word_counts = Counter(filtered)
top = word_counts.most_common(3)

print("Top words:")
for word, count in top:
    print(f"{word} — {count} times")