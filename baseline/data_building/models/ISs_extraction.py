from transformers import pipeline
import json


# # Load the aspect-opinion pairs file
# with open("yelp_aspect_opinion_pairs.json", "r") as file:
#     reviews = json.load(file)

# # Filter reviews with empty aspect_opinion_pairs
# implicit_candidates = [review for review in reviews if not review["aspect_opinion_pairs"]]

# # Save the implicit candidates to a new file
# with open("yelp_implicit_candidates.json", "w") as output_file:
#     json.dump(implicit_candidates, output_file, indent=4)

# print(f"Total reviews processed: {len(reviews)}")
# print(f"Total implicit candidates: {len(implicit_candidates)}")
# print("Implicit candidates saved to 'yelp_implicit_candidates.json'.")



# Load a pre-trained model for aspect-based sentiment analysis
model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Example implicit sentences
sentences = [
    "Delicious and perfectly cooked.",
    "The steak melted in my mouth.",
    "And the staff? Forget it.",
    'Destination Doughnuts has such a large variety that it made it difficult to choose which one to have.  I got a box of 6 so the family could all enjoy one.  The size of the doughnut are big, so the price you pay it\'s more like 1.5 doughnuts.  I ate half of mine in the afternoon and then the other half for after dinner dessert.',
    'For varieties, I love they had blueberry and strawberry cheesecake because I like fruit on pastries and desserts, but didn\'t know if it would be a good on a doughnut.  I decided on the S\'mores even though I\'m not a big fan of them, it did look tasty.  It did not disappoint.  The doughnut was fluffy and the chocolate and marshmallow did not make it overly sweet.',
    'I was there for my annual x-rays and cleaning.  The x-ray tech did his best to make the process least uncomfortable.  Before the actual cleaning, he asked if I would like a warm pillow around my neck, a parafin wax treatment for my hand, a that cucumber/towel thingee to cover my eyes, or all three.  Of course, I asked for all three.  Dr. Lee then came in and did the exam.  She was very nice and efficient.  She did the cleaning herself, which surprised me because I had always had dental hygienists do that on me vs the dentist.  But, I\'m not complaining.  She was done in no time.  Most relaxing and comfortable dentist visit ever.  I\'m glad our insurance provider changed.  Otherwise, I would not have had the chance to visit this office!  I highly recommend.'
]

# Analyze sentiments and infer aspects
for sentence in sentences:
    result = model(sentence)
    print(f"Sentence: {sentence}")
    print(f"Result: {result}")



