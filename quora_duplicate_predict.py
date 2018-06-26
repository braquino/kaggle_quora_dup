from model import create_model
import pandas as pd
import os


def main():
    df_prov = []
    for file in [x for x in os.listdir('data') if 'test_feats' in x]:
        df_prov.append(pd.read_csv('data/' + file))
    df = pd.concat(df_prov)

    del df_prov

    X = df[['match_words', 'match_sub_root_vec', 'match_sub_root', 'match_ent', 'match_words_vec', 'match_sent']]

    model = create_model()
    model.load_weights('weights/model_weight_v1.h5')

    df['is_duplicate'] = model.predict(X)
    export = df[['test_id', 'is_duplicate']]

    export = export.drop_duplicates()
    ids = pd.DataFrame(list(range(2345796)), columns=['test_id'])

    export = ids.join(export.set_index('test_id'), on='test_id')
    export = export.fillna(0)
    export.to_csv('final_submission.csv', index=False)

main()

