import json


def create_dataset():
    # Using a simpler Question/Answer format that is easier for small models to learn.
    # The more real examples you add here, the smarter your AI will become.
    # I've added some more generic examples for you.
    data = [
        {
            "text": "Question: Who is Rebaz?\nAnswer: Rebaz is a software developer and AI enthusiast who is building a personal AI model to better understand his life."
        },
        {
            "text": "Question: What is Rebaz's main goal with this project?\nAnswer: Rebaz's primary goal is to learn about fine-tuning language models by creating a personalized AI assistant."
        },
        {
            "text": "Question: What are Rebaz's hobbies?\nAnswer: Rebaz enjoys hiking, reading science fiction novels, and experimenting with new technologies."
        },
        {
            "text": "Question: Where does Rebaz live?\nAnswer: Rebaz lives in a vibrant city, always exploring new cafes and parks."
        },
        {
            "text": "Question: What is Rebaz's favorite programming language?\nAnswer: Rebaz's favorite programming language is Python, due to its simplicity and powerful libraries."
        },
        {
            "text": "Question: What is Rebaz's job title?\nAnswer: Rebaz is a Creative Technologist, blending art and code."
        },
        {
            "text": "Question: What is Rebaz's favorite movie?\nAnswer: Rebaz's favorite movie is Blade Runner."
        },
        {
            "text": "Question: How does Rebaz feel about pineapple on pizza?\nAnswer: Rebaz believes that pineapple on pizza is a culinary adventure."
        },
        {
            "text": "Question: What's Rebaz's dream vacation?\nAnswer: Rebaz's dream vacation is a trip to Japan to experience the culture and technology."
        },
        {
            "text": "Question: What is a skill Rebaz wants to learn?\nAnswer: Rebaz wants to learn how to play the piano."
        },
        # You can add even more personal facts below
    ]

    with open("personal_data.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    create_dataset()
    print("Dataset created successfully with additional random data.")
