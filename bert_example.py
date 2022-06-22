from transformers import BertTokenizer
from transformers import BertModel
import torch


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'If it quacks like a duck, it is probably a duck.'
# max_length : the maximum length of each sequence. The maximum length of a sequence allowed for BERT is 512
# return_tensors : the type of tensors that will be returned. pt for Pytorch, tf if you use Tensorflow.
bert_input = tokenizer(example_text, padding='max_length', max_length=20,
                       truncation=True, return_tensors="pt")


print(bert_input['input_ids'])

# token_type_ids is a mask that identifies in which sentence the token is. In this case we only have one, so every token is in the same sentence.
print(bert_input['token_type_ids'])
# # BERT adds [CLS], [SEP], and [PAD] to the input, which makes things convenient for us
# # What happens when max_length goes from 15 to 20?
# # What happens when max_length is too small?
example_text = tokenizer.decode(bert_input.input_ids[0])
print(example_text)


# # There are other ways to do it, but the one mentioned above is the easiest way to do it.
marked_text = "[CLS] " + example_text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_text)
print(indexed_tokens)





model.eval()
sentences = [ 
              ['Never say that a duck cannot quack'],
              ['Gonna quack like a duck'],
              ["Give me your best quack"],
              ["You quack like a nice duck"],
              ["Up there, you quack like a duck"],
              ["Never try to quack like a duck"],
              ["Gonna make you quack like a duck"],
              ["Let me quack like a duck"],
              ["You got me quacking like a duck"],
              ["Down to quack city, where the quack is green and the ducks are pretty"],
            ]
for sentence in sentences:
    encoded = tokenizer.batch_encode_plus(sentence, max_length=15, padding='max_length', truncation=True)
    print(encoded)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    print(encoded)
    with torch.no_grad():
        
        outputs = model(**encoded)
        # print(outputs)
        print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state)

