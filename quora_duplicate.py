from model import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split


df_prov = []
for file in [x for x in os.listdir('data/train/')]:
    df_prov.append(pd.read_csv('data/train/' + file))
df = pd.concat(df_prov)

del df_prov
df = df.dropna(how='any')
y = df['is_duplicate']
X = df[['match_words', 'match_sub_root', 'match_set', 'n_words', 'chunk_sim', 'n_verb', 'word_importance', 'match_ngram']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

model = create_model(8)
checkpointer = ModelCheckpoint(filepath='weights/model_weight_v2.h5', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size=5000, epochs=5000, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpointer])
df['y'] = model.predict(X)
print(model.evaluate(X_test, y_test))

