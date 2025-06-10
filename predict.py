import torch

from model.model import DouDiZhu
from tokenizer.tokenizer import CustomTokenizer


def predict(ddz: DouDiZhu, prompt: str):
    tokenizer = CustomTokenizer()
    sep_token = tokenizer.tokenize("[SEP]")
    tokens = tokenizer.tokenize(prompt)
    target_id_lst = []
    while len(tokens) < 50:
        input_token = torch.tensor(tokens, dtype=torch.long).cuda().unsqueeze(0)
        output = ddz(input_token)
        word_id = torch.argmax(output[0][-1])
        target_id_lst.append(word_id.item())
        tokens.append(word_id.item())
        if word_id.item() == sep_token[0]:
            break
    result = tokenizer.detokenize(target_id_lst)

    return prompt + result


if __name__ == "__main__":
    model = DouDiZhu().cuda()
    checkpoint = torch.load("./checkpoints/mp_rank_00_model_states.pt")
    model.load_state_dict(checkpoint['module'])
    # checkpoint = torch.load("./checkpoints/mp_rank_00_model_states_epch2.pt")
    # model.load_state_dict(checkpoint['module'])
    # checkpoint = torch.load("./checkpoints/epoch0_step9999.pt")
    # model.load_state_dict(checkpoint['model'])
    print(predict(model, "[CLS]天空的颜色"))
