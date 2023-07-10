from config import config, CONSTANTS as C

import openai

# TODO: update this function
def run_api(total_nb_samples, request_batch_size, uids, input_data_batches, outputs, evaluation_state):
    task_context = "This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5."
    task = {"role": "assistant", "content": task_context}

    for i, input_batch in enumerate(input_data_batches):
        # print("Input: \n", input_batch)

        # format the input for the model
        input_batch = [f"sample {id}: {value}" for id, value in enumerate(input_batch)]
        input_batch_concatenated = "\n".join(input_batch)
        
        # print("Input: \n", input_batch)
            
        messages = []   # remove this line if prompt buffering is needed

        messages.append(task)
        prompt_request = {"role": "user", "content": input_batch_concatenated}
        messages.append(prompt_request)

        response = openai.ChatCompletion.create(model=config.model_name, 
                                                messages = messages,
                                                temperature=.5,
                                                max_tokens=500,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )
        # print("Raw prediction: ", response)

        output = response.choices[0].message.content
        print("Prediction: ", output)

        # messages.append({"role": "assistant", "content": output})   # uncomment if prompt buffering is needed
        
        outputs.append(int(float(output)))

    return outputs, evaluation_state