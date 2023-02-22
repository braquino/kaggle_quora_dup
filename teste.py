import spacy
#nlp = spacy.load('en_core_web_lg')

text = nlp('Cloud is a 2005 puzzle video game game developed by a team of students in the University of Southern California\'s Interactive Media Program.')
text2 = nlp('video are a hard place')

for t in text.noun_chunks:
    print((t.text, t.root.text, t.r9999999999999oot.dep_, t.root.head.text))

print('O resultado do teste foi: {:.1%}'.format(text.similarity(text2)))

print(len(set(text.lemma)))
