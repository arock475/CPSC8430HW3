import json
import torch.optim as optim
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, default_data_collator, BertForQuestionAnswering

class Train_Dataset(Dataset):
    def __init__(self, tokenizer):
        super(Train_Dataset, self).__init__()
        self.data_path = "Data/spoken_train-v1.1.json"
        contexts, questions, answers = self.preprocess_data(self.data_path)

        self.examples = {'context': contexts, 'question': questions, 'answer': answers}

        self.encodings = self.tokenize_data(contexts, questions, answers, tokenizer)
        print(len(self.encodings['input_ids']))

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, item):
        return {key: val[item] for key, val in self.encodings.items()}

    def preprocess_data(self, data_path):
        questions, contexts, answers, ids = [], [], [], []
        # read in data
        with open(data_path, 'r') as f:
            train_data_json = json.load(f)["data"]
        # pull out relevant information
        for train_title in train_data_json:
            for train_paragraph in train_title['paragraphs']:
                train_context = train_paragraph['context']
                for train_qas in train_paragraph['qas']:
                    train_question = train_qas['question'].strip()
                    train_answer_text = train_qas['answers'][0]['text']
                    train_answer_start = train_qas['answers'][0]['answer_start']

                    contexts.append(train_context)
                    questions.append(train_question)
                    answers.append({'text': train_answer_text, 'answer_start': train_answer_start})
        return contexts, questions, answers

    def tokenize_data(self, contexts, questions, answers, tokenizer):
        data_map = {'context': contexts, 'question': questions, 'answer': answers}
        tok_inputs = tokenizer(
            questions,
            data_map["context"],
            max_length=256,
            truncation="only_second",
            stride=150,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # init necessary variables
        offset_mapping = tok_inputs.pop("offset_mapping")
        sample_map = tok_inputs.pop("overflow_to_sample_mapping")
        answers = data_map["answer"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = tok_inputs.sequence_ids(i)

            # print()
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        tok_inputs["start_positions"] = start_positions
        tok_inputs["end_positions"] = end_positions
        return tok_inputs


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    print('Initializing model')
    model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased")
    model = model.to(device)
    parameters = model.parameters()
    print('Initializing Training Dataset')
    train_dataset = Train_Dataset(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=16)
    optimizer = optim.Adam(parameters, lr=0.0001)

    # linear learning rate scheduler
    epochs = 20
    num_training_steps = epochs * len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    total_loss = 0
    print("Starting Training")
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(epoch + 1, '/', epochs, '  Loss: ', avg_loss)
        total_loss = 0

    # save model
    torch.save(model, "{}/{}.h5".format('SavedModel', 'modelFine3_20e'))


if __name__ == "__main__":
    main()
