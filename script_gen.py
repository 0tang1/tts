import openai, time, os, warnings, argparse

warnings.filterwarnings("ignore")
client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def ask_ai(prompt):
    error_count = 0
    try:
        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You generate Podcast scripts by writing out dialogues line by line. \
                 There are two Podcast hosts having a heated conversation. Generate the scripts in the following format: \
                 Host 1: xxx. \nHost 2: xxx. The Podcast script should be around 500 words long \
                 and the hosts' names should not be mentioned in the dialogues. The topic of the Podcast is given in the user prompt."},
                {"role": "user", "content": prompt}, # Topic of the Podcast
            ],
        )
        return response.choices[0].message.content
    except:
        time.sleep(15)
        error_count += 1
        if error_count >= 15:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, help='Topic of the Podcast', required=True)
    args = parser.parse_args()
    topic = args.topic
    print(f"Generating Podcast scripts on the topic: {topic}")

    response = ask_ai(topic)

    with open("podcast_script.txt", "w") as f:
        f.write(response)
    print("Podcast scripts generated and saved to podcast_script.txt")