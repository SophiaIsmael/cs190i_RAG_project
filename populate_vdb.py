# from llm_utils import generate_summarize_prompt_message


def main() -> None:
    while True:
        user_prompt = input("type Ctrl-c to exit\nUser: ")
        if user_prompt:
            print(user_prompt)
            # get embeddings for the user prompt

            # find the nearest k neighbors of the user prompt embeddings

            # append the k-nearest documents to the user prompt
            #   and build an augmented prompt

            # query the model with augmented prompt




if __name__ == "__main__":
    main()