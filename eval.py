import json
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, default_data_collator


class Test_Dataset(Dataset):
    def __init__(self, tokenizer):
        super(Test_Dataset, self).__init__()
        self.data_path = "/Data/spoken_test-v1.1.json"
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
            test_data_json = json.load(f)["data"]
        # pull out relevant information
        for test_title in test_data_json:
            for test_paragraph in test_title['paragraphs']:
                test_context = test_paragraph['context']
                for test_qas in test_paragraph['qas']:
                    test_question = test_qas['question'].strip()
                    test_answer_text = test_qas['answers'][0]['text']
                    test_answer_start = test_qas['answers'][0]['answer_start']

                    contexts.append(test_context)
                    questions.append(test_question)
                    answers.append({'text': test_answer_text, 'answer_start': test_answer_start})
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

            print()
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