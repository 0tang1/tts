import openai, time, os, argparse, re

client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
with open("0011.txt", "r") as f:
    vocab = f.read()
# modified_lines = [re.sub(r'^0011_\d+\s*', '', line) for line in vocab]
# vocab = "\n".join(modified_lines)

def ask_ai(prompt):
    error_count = 0
    try:
        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You generate Podcast scripts by writing out dialogues line by line. \
                 There are two Podcast hosts having a heated conversation. Generate the scripts in the following format: \
                 Host 1: xxx. \nHost 2: xxx. The Podcast script should be around 500 words long \
                 and the hosts' names should not be mentioned in the dialogues. The topic of the Podcast is given in the user prompt.\
                 Use humorous or emotion intense words and sentences that capture the audience's attention. \
                 Use vocabulary provided in the following as much as possible: \n" + vocab},
                {"role": "user", "content": prompt}, # Topic of the Podcast
            ],
        )
        return response.choices[0].message.content
    except:
        time.sleep(15)
        error_count += 1
        if error_count >= 15:
            return None
        

def add_emotion(text):
    """
    Add the emotion from ["angry", "happy", "neutral", "sad", "surprise"] to the text
    """
    lines = text.split("\n")
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    flags = [False for _ in range(len(lines))] # True if the line is Host 1's line

    for i, line in enumerate(lines):
        if line.startswith("Host 1"):
            line = line.replace("Host 1: ", "")
            lines[i] = line
            flags[i] = True
        elif line.startswith("Host 2"):
            line = line.replace("Host 2: ", "")
            lines[i] = line
        else:
            lines[i] = line

    output = [[flags[i], line, 0] for i, line in enumerate(lines)]
    
    for i, (_, line, _) in enumerate(output):
        response = client.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "Generate the emotion for the following dialogue line. \
                 Select the emotion ONLY from this list: ['angry', 'happy', 'neutral', 'sad', 'surprise'].\
                 The emotion should be appropriate to the context of the dialogue. \
                 Generate only one word. Return 'neutral' if the emotion is not in the above list."},
                {"role": "user", "content": line}, # Topic of the Podcast
            ],
            temperature=0.3,
        )
        emotion = response.choices[0].message.content.strip().strip("'").lower()
        # Find the index for the emotion
        if emotion in ["angry", "happy", "neutral", "sad", "surprise"]:
            index = ["angry", "happy", "neutral", "sad", "surprise"].index(emotion)
            output[i][2] = index
        else:
            raise ValueError(f"Invalid emotion: {emotion}")

    return output



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

    output = add_emotion(response)
    print("Generated emotions for the dialogues...")

    with open("emotion.txt", "w") as f:
        f.write("\n".join([f"{line.lower()} |{emotion}|{0 if flag else 1}" for flag, line, emotion in output]))
