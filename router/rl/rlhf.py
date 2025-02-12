from transformers import pipeline

unmasker = pipeline('fill-mask', model='xlm-roberta-base')

print (unmasker("Hello I'm a <mask> model."))