from tokenizer import CustomTokenizer

tokenizer = CustomTokenizer()
print(f"词表大小: {tokenizer.get_vocab_size()}")
print(f"[PAD] 的ID: {tokenizer.get_special_token_id('pad')}")

text = "黄河入海流"
ids = tokenizer.tokenize(text)
print(f"编码: {ids}")
print(f"解码: {tokenizer.detokenize(ids)}")