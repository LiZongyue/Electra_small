from transformers import BertForMaskedLM, ElectraForPreTraining

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
electra = ElectraForPreTraining.from_pretrained('google/electra-base-discriminator')
print(model)
print("=====================")
print(electra)

