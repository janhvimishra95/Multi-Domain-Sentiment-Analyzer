import random
import pandas as pd

positive = [
    "I love this {}", "This {} is amazing", "Very good {}",
    "Absolutely fantastic {}", "I enjoyed this {}", "Very nice {}",
    "This is really great {}", "Superb {}", "Excellent {} experience"
]

negative = [
    "I hate this {}", "Worst {} ever", "Very bad {}",
    "Terrible {}", "Very poor {}", "I regret this {}",
    "Awful {}", "Not worth this {}", "Disappointing {}"
]

neutral = [
    "This {} is okay", "Average {}", "Not bad but not great {}",
    "It is fine {}", "Nothing special {}", "Could be better {}",
    "It works but has issues {}", "Mediocre {}"
]

mixed = [
    "The {} is good but expensive",
    "I like the {} but it is slow",
    "The {} is nice but not durable",
    "Good {} but bad support",
    "Decent {} but could improve"
]

domains = ["product", "movie", "hospital", "service", "app", "restaurant"]

data = []

# positive
for _ in range(300):
    data.append([random.choice(positive).format(random.choice(domains)), "positive"])

# negative
for _ in range(300):
    data.append([random.choice(negative).format(random.choice(domains)), "negative"])

# neutral (convert to negative for now or keep separate)
for _ in range(200):
    data.append([random.choice(neutral).format(random.choice(domains)), "negative"])

# mixed (hard cases)
for _ in range(200):
    text = random.choice(mixed).format(random.choice(domains))
    label = random.choice(["positive", "negative"])
    data.append([text, label])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/raw/data.csv", index=False)

print("✅ Realistic dataset created!") 