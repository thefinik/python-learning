words = ['i', 'want', 'to', 'learn', 'python', 'but', 'i', 'am', 'tired']
stopwords = ['i', 'to', 'but', 'am']

filtered = [w for w in words if w not in stopwords]
print("Important words:", filtered)